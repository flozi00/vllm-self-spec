from __future__ import annotations

import argparse
import atexit
import contextlib
import json
import logging
import math
import os
import random
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


_TERMINATION_REQUESTED = False
logger = logging.getLogger("vllm_colocate.lora_trainer")


def _handle_termination_signal(_signum: int, _frame: Any | None) -> None:
    global _TERMINATION_REQUESTED
    _TERMINATION_REQUESTED = True


def _install_signal_handlers() -> None:
    try:
        signal.signal(signal.SIGTERM, _handle_termination_signal)
        signal.signal(signal.SIGINT, _handle_termination_signal)
    except ValueError:
        return


@dataclass(frozen=True)
class LoraTrainerConfig:
    model_name: str
    data_path: Path
    checkpoint_dir: Path
    adapter_name: str
    task_mix: str = "mixed"
    device: str = "cuda"
    parallel_mode: str = "auto"
    max_steps: int = 0
    train_epochs: float = 1.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_seq_len: int = 2048
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    dpo_beta: float = 0.1
    kto_beta: float = 0.1
    kto_desirable_weight: float = 1.0
    kto_undesirable_weight: float = 1.0
    loss_vocab_sample_size: int = 0
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    include_expert_lora: bool = False
    quantization: str = "auto"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    save_optimizer_state: bool = True
    checkpoint_every: int = 16
    keep_last_checkpoints: int = 2
    stop_file: Path | None = None
    metrics_path: Path | None = None
    run_id: str = ""
    log_every: int = 1


class MissingTrainingDependency(RuntimeError):
    pass


_MODELOPT_NVFP4_STATE_ATTR = "_vllm_colocate_modelopt_nvfp4_state"
_FOUROVERSIX_DEQUANTIZED_WEIGHT_CACHE_ATTR = (
    "_vllm_colocate_fouroversix_dequantized_weight_cache"
)
_FOUROVERSIX_FORCE_DEQUANTIZED_FORWARD_ATTR = (
    "_vllm_colocate_fouroversix_force_dequantized_forward"
)
_MODELOPT_NVFP4_MODULE_NAME_ATTR = "_vllm_colocate_modelopt_nvfp4_module_name"
_FOUROVERSIX_DEQUANTIZED_CACHE_BYTES = 0
_MODELOPT_NVFP4_CPU_CACHE_BYTES = 0


@dataclass
class _ModelOptNvfp4State:
    store: Any
    weight_key: str
    scale_key: str
    global_scale_key: str
    tensors: dict[str, Any] = field(default_factory=dict)

    def tensor(self, role: str) -> Any:
        key = {
            "weight": self.weight_key,
            "scale": self.scale_key,
            "global_scale": self.global_scale_key,
        }[role]
        if role not in self.tensors:
            self.tensors[role] = self.store.tensor(key)
        return self.tensors[role]


class _SafetensorsKeyStore:
    def __init__(self, snapshot_path: Path):
        self.snapshot_path = snapshot_path
        self.weight_map: dict[str, str] = {}
        index_path = snapshot_path / "model.safetensors.index.json"
        if index_path.exists():
            data = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map.update(data.get("weight_map") or {})
        if not self.weight_map:
            from safetensors import safe_open

            for safetensors_path in sorted(snapshot_path.glob("*.safetensors")):
                with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        self.weight_map.setdefault(key, safetensors_path.name)

    def __contains__(self, key: str) -> bool:
        return key in self.weight_map

    def tensor(self, key: str) -> Any:
        from safetensors import safe_open

        filename = self.weight_map[key]
        path = self.snapshot_path / filename
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)


def run_training(config: LoraTrainerConfig) -> int:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _configure_logging()
    _configure_torch_matmul_precision(torch)
    _install_transformers_remote_code_compat()
    _install_signal_handlers()
    run_started = time.time()
    rank = int(os.getenv("RANK", "0") or 0)
    local_rank = int(os.getenv("LOCAL_RANK", "0") or 0)
    primary = rank == 0

    rows = _load_training_rows(config.data_path)
    rows_by_task = {
        task: [row for row in rows if row.get("task") == task]
        for task in ("sft", "dpo", "kto")
    }
    if config.task_mix in rows_by_task:
        rows_by_task = {
            task: (task_rows if task == config.task_mix else [])
            for task, task_rows in rows_by_task.items()
        }
    sft_rows = rows_by_task["sft"]
    dpo_rows = rows_by_task["dpo"]
    kto_rows = rows_by_task["kto"]
    if not sft_rows and not dpo_rows and not kto_rows:
        logger.info("LoRA trainer found no SFT/DPO/KTO rows at %s", config.data_path)
        _write_metric(config, {"event": "not_enough_examples", "examples": 0})
        return 0

    logger.info(
        "LoRA trainer starting run_id=%s model=%s data_path=%s checkpoint_dir=%s sft_rows=%d dpo_rows=%d kto_rows=%d parallel_mode=%s rank=%d local_rank=%d",
        config.run_id or "-",
        config.model_name,
        config.data_path,
        config.checkpoint_dir,
        len(sft_rows),
        len(dpo_rows),
        len(kto_rows),
        config.parallel_mode,
        rank,
        local_rank,
    )
    _write_metric(
        config,
        {
            "event": "run_start",
            "trainer": "lora",
            "model_name": config.model_name,
            "data_path": str(config.data_path),
            "checkpoint_dir": str(config.checkpoint_dir),
            "adapter_name": config.adapter_name,
            "sft_rows": len(sft_rows),
            "dpo_rows": len(dpo_rows),
            "kto_rows": len(kto_rows),
            "task_mix": config.task_mix,
            "parallel_mode": config.parallel_mode,
            "rank": rank,
            "local_rank": local_rank,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_steps": config.max_steps,
            "train_epochs": config.train_epochs,
            "max_seq_len": config.max_seq_len,
            "learning_rate": config.learning_rate,
            "dpo_beta": config.dpo_beta,
            "kto_beta": config.kto_beta,
            "kto_desirable_weight": config.kto_desirable_weight,
            "kto_undesirable_weight": config.kto_undesirable_weight,
            "loss_vocab_sample_size": config.loss_vocab_sample_size,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "lora_target_modules": list(config.lora_target_modules),
            "include_expert_lora": config.include_expert_lora,
            "quantization": config.quantization,
            "torch_dtype": config.torch_dtype,
        },
    )
    if _stop_requested(config.stop_file):
        _write_metric(config, {"event": "stopped_before_start", "trainer": "lora"})
        return 0

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code, use_fast=True
    )
    _configure_tokenizer(tokenizer)
    sft_examples = [
        example
        for row in sft_rows
        for example in [_build_sft_example(tokenizer, row, config.max_seq_len)]
        if example is not None
    ]
    dpo_examples = [
        example
        for row in dpo_rows
        for example in [_build_dpo_example(tokenizer, row, config.max_seq_len)]
        if example is not None
    ]
    kto_examples = _build_kto_examples(tokenizer, kto_rows, config.max_seq_len)
    if not sft_examples and not dpo_examples and not kto_examples:
        logger.info("LoRA trainer could not tokenize any SFT/DPO/KTO examples")
        _write_metric(config, {"event": "not_enough_tokenized_examples", "examples": 0})
        return 0

    try:
        model_kwargs = _model_load_kwargs(config)
    except MissingTrainingDependency as exc:
        message = str(exc)
        logger.error("LoRA trainer dependency check failed: %s", message)
        _write_metric(
            config,
            {
                "event": "missing_training_dependency",
                "trainer": "lora",
                "error": message,
                "quantization": config.quantization,
            },
        )
        return 2
    logger.info("LoRA trainer loading base model with kwargs=%s", model_kwargs)
    try:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    except Exception as exc:
        if not _is_missing_training_dependency_exception(exc):
            raise
        message = str(exc)
        logger.error("LoRA trainer model load dependency failed: %s", message)
        _write_metric(
            config,
            {
                "event": "missing_training_dependency",
                "trainer": "lora",
                "error": message,
                "quantization": config.quantization,
            },
        )
        return 2
    modelopt_nvfp4_modules = _attach_modelopt_nvfp4_sidecar_tensors(
        model, config.model_name
    )
    if modelopt_nvfp4_modules:
        logger.info(
            "LoRA trainer attached ModelOpt NVFP4 sidecar tensors to %d modules",
            modelopt_nvfp4_modules,
        )
    if hasattr(model, "config"):
        model.config.use_cache = False

    logger.info(
        "LoRA trainer injecting modules targets=%s include_expert_lora=%s",
        ",".join(config.lora_target_modules),
        config.include_expert_lora,
    )
    wrappers = _inject_lora_modules(
        model,
        target_modules=config.lora_target_modules,
        include_expert_lora=config.include_expert_lora,
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )
    logger.info("LoRA trainer injected %d LoRA modules", len(wrappers))
    if not wrappers:
        raise RuntimeError(
            "No LoRA target modules were found. Set VLLM_COLOCATE_LORA_TARGET_MODULES "
            "for this model architecture."
        )
    if _lora_targets_need_backbone_grad_checkpointing(config.lora_target_modules) and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            logger.debug("Could not enable gradient checkpointing", exc_info=True)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("LoRA injection produced no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    step = _restore_latest(config, model, optimizer if config.save_optimizer_state else None)

    first_device = _first_parameter_device(model, config.device, torch)
    if "device_map" not in model_kwargs:
        model.to(first_device)
    model.train()

    planned_steps = _planned_steps(
        config, len(sft_examples) + len(dpo_examples) + len(kto_examples)
    )
    available_tasks = [
        task
        for task, examples in (
            ("sft", sft_examples),
            ("dpo", dpo_examples),
            ("kto", kto_examples),
        )
        if examples
    ]
    if step > 0:
        logger.info("LoRA trainer restored checkpoint at global_step=%d", step)
    _write_metric(
        config,
        {
            "event": "ready",
            "trainer": "lora",
            "sft_examples": len(sft_examples),
            "dpo_examples": len(dpo_examples),
            "kto_examples": len(kto_examples),
            "lora_modules": sorted(wrappers),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
            "restored_step": step,
            "planned_steps": planned_steps,
            "device": str(first_device),
        },
    )

    completed_steps = 0
    last_loss = None
    accum = max(1, config.gradient_accumulation_steps)
    optimizer.zero_grad(set_to_none=True)
    for local_step in range(planned_steps):
        if _TERMINATION_REQUESTED or _stop_requested(config.stop_file):
            logger.info("LoRA trainer stopping at local_step=%d", local_step)
            break
        task = _task_for_step(local_step, available_tasks)
        loss_value = 0.0
        for accum_index in range(accum):
            if task == "dpo":
                batch = _dpo_batch_for_step(
                    dpo_examples,
                    step + local_step,
                    accum_index,
                    config.batch_size,
                    tokenizer.pad_token_id,
                    torch,
                    first_device,
                )
                loss = _dpo_loss(
                    model,
                    batch,
                    beta=config.dpo_beta,
                    loss_vocab_sample_size=config.loss_vocab_sample_size,
                    torch=torch,
                    F=F,
                )
            elif task == "kto":
                batch = _kto_batch_for_step(
                    kto_examples,
                    step + local_step,
                    accum_index,
                    config.batch_size,
                    tokenizer.pad_token_id,
                    torch,
                    first_device,
                )
                loss = _kto_loss(
                    model,
                    batch,
                    beta=config.kto_beta,
                    desirable_weight=config.kto_desirable_weight,
                    undesirable_weight=config.kto_undesirable_weight,
                    loss_vocab_sample_size=config.loss_vocab_sample_size,
                    torch=torch,
                    F=F,
                )
            else:
                batch = _sft_batch_for_step(
                    sft_examples,
                    step + local_step,
                    accum_index,
                    config.batch_size,
                    tokenizer.pad_token_id,
                    torch,
                    first_device,
                )
                loss = _sft_loss(
                    model,
                    batch,
                    loss_vocab_sample_size=config.loss_vocab_sample_size,
                    torch=torch,
                    F=F,
                )
            if not bool(torch.isfinite(loss.detach()).all().cpu()):
                raise RuntimeError(
                    f"LoRA trainer produced non-finite {task} loss "
                    f"at local_step={local_step} accum_index={accum_index}"
                )
            (loss / accum).backward()
            loss_value += float(loss.detach().float().cpu())
        sanitized_gradients = _sanitize_nonfinite_gradients(trainable_parameters, torch)
        if sanitized_gradients:
            logger.warning(
                "LoRA trainer sanitized %d non-finite gradient tensor(s) "
                "at local_step=%d task=%s",
                sanitized_gradients,
                local_step,
                task,
            )
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, config.max_grad_norm)
        optimizer.step()
        sanitized_optimizer_states = _sanitize_nonfinite_optimizer_state(optimizer, torch)
        if sanitized_optimizer_states:
            logger.warning(
                "LoRA trainer sanitized %d non-finite optimizer state tensor(s) "
                "at local_step=%d task=%s",
                sanitized_optimizer_states,
                local_step,
                task,
            )
        optimizer.zero_grad(set_to_none=True)
        completed_steps += 1
        global_step = step + completed_steps
        last_loss = loss_value / accum
        if primary and config.log_every > 0 and (
            completed_steps == 1 or completed_steps % config.log_every == 0
        ):
            elapsed = time.time() - run_started
            progress = completed_steps / max(1, planned_steps)
            logger.info(
                "LoRA trainer step %d/%d task=%s loss=%.4f elapsed=%.1fs",
                completed_steps,
                planned_steps,
                task,
                last_loss,
                elapsed,
            )
            _write_metric(
                config,
                {
                    "event": "step",
                    "trainer": "lora",
                    "step": global_step,
                    "local_step": completed_steps,
                    "max_steps": planned_steps,
                    "progress": progress,
                    "task": task,
                    "loss": last_loss,
                    "elapsed_seconds": elapsed,
                },
            )
        if primary and config.checkpoint_every > 0 and global_step % config.checkpoint_every == 0:
            _save(config, model, optimizer if config.save_optimizer_state else None, global_step)

    if primary and completed_steps > 0:
        _save(
            config,
            model,
            optimizer if config.save_optimizer_state else None,
            step + completed_steps,
        )
    _write_metric(
        config,
        {
            "event": "run_complete",
            "trainer": "lora",
            "step": step + completed_steps,
            "completed_steps": completed_steps,
            "max_steps": planned_steps,
            "loss": last_loss,
            "stopped": _TERMINATION_REQUESTED or _stop_requested(config.stop_file),
            "elapsed_seconds": time.time() - run_started,
        },
    )
    return 0


def _load_training_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("task") in {"sft", "dpo", "kto"}:
                    rows.append(row)
    except FileNotFoundError:
        return []
    return rows


def _configure_tokenizer(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos = getattr(tokenizer, "eos_token", None)
        if eos is not None:
            tokenizer.pad_token = eos


def _build_sft_example(
    tokenizer: Any, row: dict[str, Any], max_seq_len: int
) -> dict[str, list[int]] | None:
    prompt_text = ""
    full_text = ""
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        prompt_messages, full_messages = _split_prompt_completion_messages(messages)
        prompt_text = _render_messages(tokenizer, prompt_messages, add_generation_prompt=True)
        full_text = _render_messages(tokenizer, full_messages, add_generation_prompt=False)
    elif row.get("prompt") is not None and row.get("completion") is not None:
        prompt_text = str(row.get("prompt") or "")
        full_text = prompt_text + str(row.get("completion") or "")
    elif row.get("text") is not None:
        full_text = str(row.get("text") or "")
    if not full_text:
        return None
    full_ids = _tokenize_text(tokenizer, full_text, max_seq_len=max_seq_len)
    if not full_ids:
        return None
    prompt_len = 0
    if prompt_text:
        prompt_len = len(_tokenize_text(tokenizer, prompt_text, max_seq_len=max_seq_len))
        prompt_len = min(prompt_len, max(0, len(full_ids) - 1))
    labels = list(full_ids)
    for index in range(prompt_len):
        labels[index] = -100
    if not any(label != -100 for label in labels):
        return None
    return {"input_ids": full_ids, "labels": labels}


def _build_dpo_example(
    tokenizer: Any, row: dict[str, Any], max_seq_len: int
) -> dict[str, dict[str, list[int]]] | None:
    prompt_text = _prompt_text(tokenizer, row)
    chosen_text = _completion_text(tokenizer, row.get("chosen"))
    rejected_text = _completion_text(tokenizer, row.get("rejected"))
    if not prompt_text or not chosen_text or not rejected_text:
        return None
    chosen = _sequence_with_prompt_mask(tokenizer, prompt_text, chosen_text, max_seq_len)
    rejected = _sequence_with_prompt_mask(tokenizer, prompt_text, rejected_text, max_seq_len)
    if chosen is None or rejected is None:
        return None
    return {"chosen": chosen, "rejected": rejected}


def _build_kto_examples(
    tokenizer: Any, rows: list[dict[str, Any]], max_seq_len: int
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for row in rows:
        prompt_text = _prompt_text(tokenizer, row)
        completion_text = _completion_text(tokenizer, row.get("completion"))
        label = row.get("label")
        if label is None:
            label = row.get("desirable")
        if not prompt_text or not completion_text or label is None:
            continue
        target = _sequence_with_prompt_mask(
            tokenizer, prompt_text, completion_text, max_seq_len
        )
        if target is None:
            continue
        parsed.append(
            {
                "prompt_text": prompt_text,
                "completion_text": completion_text,
                "target": target,
                "desirable": bool(label),
            }
        )
    examples: list[dict[str, Any]] = []
    for index, item in enumerate(parsed):
        example: dict[str, Any] = {
            "target": item["target"],
            "desirable": item["desirable"],
        }
        if len(parsed) > 1:
            # KL baseline sequences pair each prompt with an unrelated
            # completion, following the mismatched-pair estimate from the
            # KTO paper.
            other = parsed[(index + 1) % len(parsed)]
            kl_sequence = _sequence_with_prompt_mask(
                tokenizer,
                item["prompt_text"],
                other["completion_text"],
                max_seq_len,
            )
            if kl_sequence is not None:
                example["kl"] = kl_sequence
        examples.append(example)
    return examples


def _render_messages(
    tokenizer: Any, messages: list[dict[str, Any]], *, add_generation_prompt: bool
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        return "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in messages
        )


def _split_prompt_completion_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for index in range(len(messages) - 1, -1, -1):
        if str(messages[index].get("role") or "") == "assistant":
            return messages[:index], messages[: index + 1]
    return [], messages


def _prompt_text(tokenizer: Any, row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if prompt is not None:
        return str(prompt)
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return _render_messages(tokenizer, messages, add_generation_prompt=True)
    return ""


def _completion_text(tokenizer: Any, value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return _render_messages(tokenizer, value, add_generation_prompt=False)
    if isinstance(value, dict):
        if isinstance(value.get("messages"), list):
            return _render_messages(tokenizer, value["messages"], add_generation_prompt=False)
        if value.get("content") is not None:
            return str(value.get("content") or "")
    return ""


def _tokenize_text(tokenizer: Any, text: str, *, max_seq_len: int) -> list[int]:
    token_ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
    return [int(token_id) for token_id in token_ids][-max_seq_len:]


def _sequence_with_prompt_mask(
    tokenizer: Any, prompt_text: str, completion_text: str, max_seq_len: int
) -> dict[str, list[int]] | None:
    prompt_ids = _tokenize_text(tokenizer, prompt_text, max_seq_len=max_seq_len)
    full_ids = _tokenize_text(
        tokenizer, prompt_text + completion_text, max_seq_len=max_seq_len
    )
    if not full_ids:
        return None
    prompt_len = min(len(prompt_ids), max(0, len(full_ids) - 1))
    labels = list(full_ids)
    for index in range(prompt_len):
        labels[index] = -100
    if not any(label != -100 for label in labels):
        return None
    return {"input_ids": full_ids, "labels": labels}


def _model_load_kwargs(config: LoraTrainerConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    dtype = _torch_dtype_from_name(config.torch_dtype)
    if dtype is not None:
        kwargs["dtype"] = dtype
    quant_config = _training_quantization_config(config)
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    if _use_device_map(config):
        kwargs["device_map"] = "auto"
    return kwargs


def _install_transformers_remote_code_compat() -> None:
    try:
        from transformers.utils import generic
    except Exception:
        return

    if not hasattr(generic, "OutputRecorder"):
        try:
            from transformers.utils.output_capturing import OutputRecorder
        except Exception:
            class OutputRecorder:
                def __init__(
                    self,
                    target_class: Any = None,
                    index: int = 0,
                    layer_name: str | None = None,
                    class_name: str | None = None,
                    capture_initial_hidden_state: bool = True,
                ) -> None:
                    self.target_class = target_class
                    self.index = index
                    self.layer_name = layer_name
                    self.class_name = class_name
                    self.capture_initial_hidden_state = capture_initial_hidden_state

        generic.OutputRecorder = OutputRecorder

    if not hasattr(generic, "check_model_inputs"):
        generic.check_model_inputs = lambda function: function

    _install_fouroversix_linear_compat()
    _install_masking_utils_compat()
    _install_rotary_embedding_method_compat()

    try:
        from transformers import modeling_rope_utils
    except Exception:
        return
    if "default" not in getattr(modeling_rope_utils, "ROPE_INIT_FUNCTIONS", {}):
        modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _install_masking_utils_compat() -> None:
    try:
        import inspect
        from transformers import masking_utils
    except Exception:
        return

    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        function = getattr(masking_utils, name, None)
        if not callable(function) or getattr(
            function, "_vllm_colocate_masking_compat", False
        ):
            continue
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            signature = None
        accepts_var_kwargs = (
            signature is not None
            and any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        )
        accepted_kwargs = set(signature.parameters) if signature is not None else set()

        def _compat_mask_function(
            *args: Any,
            __function: Any = function,
            __accepted_kwargs: set[str] = accepted_kwargs,
            __accepts_var_kwargs: bool = accepts_var_kwargs,
            **kwargs: Any,
        ) -> Any:
            if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            if not __accepts_var_kwargs and __accepted_kwargs:
                kwargs = {
                    key: value
                    for key, value in kwargs.items()
                    if key in __accepted_kwargs
                }
            return __function(*args, **kwargs)

        _compat_mask_function.__name__ = getattr(function, "__name__", name)
        _compat_mask_function._vllm_colocate_masking_compat = True
        setattr(masking_utils, name, _compat_mask_function)


def _install_fouroversix_linear_compat() -> None:
    try:
        from fouroversix.model.modules import linear
    except Exception:
        return
    cls = getattr(linear, "FourOverSixLinear", None)
    original_forward = getattr(cls, "forward", None)
    if not callable(original_forward) or getattr(
        original_forward, "_vllm_colocate_contiguous_input", False
    ):
        return

    def _compat_forward(self: Any, input: Any, *args: Any, **kwargs: Any) -> Any:
        if hasattr(input, "is_contiguous") and not input.is_contiguous():
            input = input.contiguous()
        if getattr(input, "numel", lambda: 1)() == 0:
            features = _linear_features(self)
            if features is not None:
                return input.new_empty(*input.shape[:-1], features[1])
        if getattr(self, _FOUROVERSIX_FORCE_DEQUANTIZED_FORWARD_ATTR, False):
            return _fouroversix_dequantized_linear_forward(
                self,
                input,
                RuntimeError("using cached dequantized FourOverSix training forward"),
            )
        if (
            getattr(self, _MODELOPT_NVFP4_STATE_ATTR, None) is not None
            and getattr(input, "ndim", 0) == 2
        ):
            if (
                not getattr(input, "requires_grad", False)
                and _env_bool_any(("VLLM_COLOCATE_MODELOPT_NVFP4_NATIVE_FIRST",), True)
            ):
                try:
                    output = original_forward(self, input.unsqueeze(0), *args, **kwargs)
                    return output.squeeze(0) if hasattr(output, "squeeze") else output
                except ValueError:
                    logger.debug(
                        "Falling back to ModelOpt NVFP4 sidecar forward for %s",
                        getattr(self, _MODELOPT_NVFP4_MODULE_NAME_ATTR, "-"),
                        exc_info=True,
                    )
            return _fouroversix_dequantized_linear_forward(
                self,
                input,
                RuntimeError("using ModelOpt NVFP4 sidecar FourOverSix forward"),
            )
        if getattr(input, "ndim", 0) == 2:
            try:
                output = original_forward(self, input.unsqueeze(0), *args, **kwargs)
                return output.squeeze(0) if hasattr(output, "squeeze") else output
            except ValueError as exc:
                if "same inner dimension" not in str(exc):
                    raise
                setattr(self, _FOUROVERSIX_FORCE_DEQUANTIZED_FORWARD_ATTR, True)
                return _fouroversix_dequantized_linear_forward(self, input, exc)
        if (
            getattr(input, "requires_grad", False)
            and (getattr(input, "ndim", 0) < 3 or int(input.shape[0]) != 1)
        ):
            return _fouroversix_dequantized_linear_forward(
                self,
                input,
                RuntimeError("using dequantized FourOverSix training forward"),
            )
        try:
            return original_forward(self, input, *args, **kwargs)
        except ValueError as exc:
            if "same inner dimension" not in str(exc):
                raise
            setattr(self, _FOUROVERSIX_FORCE_DEQUANTIZED_FORWARD_ATTR, True)
            return _fouroversix_dequantized_linear_forward(self, input, exc)

    _compat_forward._vllm_colocate_contiguous_input = True
    cls.forward = _compat_forward


def _fouroversix_dequantized_linear_forward(
    module: Any,
    input: Any,
    original_error: Exception,
) -> Any:
    try:
        import torch
    except Exception:
        raise original_error

    flat_input = input.reshape(-1, input.shape[-1]).contiguous()
    modelopt_state = getattr(module, _MODELOPT_NVFP4_STATE_ATTR, None)
    if modelopt_state is not None:
        if (
            not getattr(flat_input, "requires_grad", False)
            and _env_bool_any(("VLLM_COLOCATE_MODELOPT_NVFP4_CPU_FIRST",), False)
        ):
            cpu_output = _modelopt_nvfp4_cpu_linear_forward(
                module,
                modelopt_state,
                input,
                flat_input,
            )
            if cpu_output is not None:
                return cpu_output
        weight_tensor = _cached_modelopt_nvfp4_weight(
            module,
            modelopt_state,
            device=flat_input.device,
            dtype=flat_input.dtype,
        )
        if weight_tensor is None or flat_input.shape[-1] != weight_tensor.shape[-1]:
            direct_weight_tensor = _uncached_modelopt_nvfp4_weight(
                modelopt_state,
                device=flat_input.device,
                dtype=flat_input.dtype,
            )
            if direct_weight_tensor is not None:
                weight_tensor = direct_weight_tensor
        if weight_tensor is None or flat_input.shape[-1] != weight_tensor.shape[-1]:
            direct_weight_tensor = _force_modelopt_nvfp4_sidecar_weight(
                modelopt_state,
                device=flat_input.device,
                dtype=flat_input.dtype,
            )
            if direct_weight_tensor is not None:
                weight_tensor = direct_weight_tensor
        if weight_tensor is not None and flat_input.shape[-1] == weight_tensor.shape[-1]:
            output = flat_input @ weight_tensor.t()
            output = output.reshape(*input.shape[:-1], output.shape[-1])
            bias = getattr(module, "bias", None)
            if bias is not None:
                output = output + bias.to(device=output.device, dtype=output.dtype)
            return output
        cpu_output = _modelopt_nvfp4_cpu_linear_forward(
            module,
            modelopt_state,
            input,
            flat_input,
        )
        if cpu_output is not None:
            return cpu_output
        if weight_tensor is not None:
            _log_fouroversix_fallback_mismatch_once(
                module,
                "modelopt_sidecar_shape_mismatch",
                input_shape=tuple(input.shape),
                weight_shape=tuple(weight_tensor.shape),
            )
        else:
            _log_fouroversix_fallback_mismatch_once(
                module,
                "modelopt_sidecar_unavailable",
                input_shape=tuple(input.shape),
                weight_shape=None,
            )
        raise RuntimeError(
            "FourOverSix ModelOpt NVFP4 sidecar fallback cannot align input and weight "
            f"shapes: input={tuple(input.shape)} "
            f"weight={tuple(weight_tensor.shape) if weight_tensor is not None else None} "
            f"module={getattr(module, _MODELOPT_NVFP4_MODULE_NAME_ATTR, '-')!r} "
            f"weight_key={modelopt_state.weight_key!r}"
        ) from original_error

    try:
        from fouroversix.matmul.pytorch import dequantize
    except Exception:
        raise original_error

    weight = module.quantized_weight()
    if isinstance(weight, torch.nn.Parameter):
        weight_tensor = weight.data
    elif isinstance(weight, torch.Tensor):
        weight_tensor = weight
    else:
        cache_key = ("fouroversix", str(input.device), str(input.dtype))
        cache = getattr(module, _FOUROVERSIX_DEQUANTIZED_WEIGHT_CACHE_ATTR, None)
        if cache is None:
            cache = {}
            setattr(module, _FOUROVERSIX_DEQUANTIZED_WEIGHT_CACHE_ATTR, cache)
        weight_tensor = cache.get(cache_key)
        if weight_tensor is None:
            weight_tensor = dequantize(weight, dtype=input.dtype)
            _maybe_cache_dequantized_weight(cache, cache_key, weight_tensor)
    weight_tensor = weight_tensor.to(device=input.device, dtype=input.dtype)
    if flat_input.shape[-1] == weight_tensor.shape[-1]:
        output = flat_input @ weight_tensor.t()
    elif flat_input.shape[-1] == weight_tensor.shape[0]:
        output = flat_input @ weight_tensor
    else:
        raise RuntimeError(
            "FourOverSix dequantized fallback cannot align input and weight shapes: "
            f"input={tuple(input.shape)} weight={tuple(weight_tensor.shape)} "
            f"module={getattr(module, _MODELOPT_NVFP4_MODULE_NAME_ATTR, '-')!r} "
            f"has_modelopt_sidecar={modelopt_state is not None}"
        ) from original_error
    output = output.reshape(*input.shape[:-1], output.shape[-1])
    bias = getattr(module, "bias", None)
    if bias is not None:
        output = output + bias.to(device=output.device, dtype=output.dtype)
    return output


def _modelopt_nvfp4_cpu_linear_forward(
    module: Any,
    state: _ModelOptNvfp4State,
    input: Any,
    flat_input: Any,
) -> Any | None:
    global _MODELOPT_NVFP4_CPU_CACHE_BYTES

    try:
        import torch
    except Exception:
        return None

    try:
        cache_key = ("modelopt_cpu", state.weight_key)
        cache = getattr(module, _FOUROVERSIX_DEQUANTIZED_WEIGHT_CACHE_ATTR, None)
        if cache is None:
            cache = {}
            setattr(module, _FOUROVERSIX_DEQUANTIZED_WEIGHT_CACHE_ATTR, cache)
        weight_tensor = cache.get(cache_key)
        if weight_tensor is None:
            weight_tensor = _force_modelopt_nvfp4_sidecar_weight(
                state,
                device="cpu",
                dtype=torch.float32,
            )
            if weight_tensor is not None:
                weight_tensor = weight_tensor.detach()
                tensor_bytes = _tensor_nbytes(weight_tensor)
                max_cache_bytes = _env_int(
                    "VLLM_COLOCATE_MODELOPT_CPU_DEQUANT_CACHE_MAX_BYTES",
                    4 * 1024 * 1024 * 1024,
                )
                if (
                    tensor_bytes > 0
                    and max_cache_bytes > 0
                    and _MODELOPT_NVFP4_CPU_CACHE_BYTES + tensor_bytes
                    <= max_cache_bytes
                ):
                    cache[cache_key] = weight_tensor
                    _MODELOPT_NVFP4_CPU_CACHE_BYTES += tensor_bytes
        if weight_tensor is None or flat_input.shape[-1] != weight_tensor.shape[-1]:
            return None
        input_cpu = flat_input.to(device="cpu", dtype=torch.float32)
        output = input_cpu @ weight_tensor.t()
        output = output.reshape(*input.shape[:-1], output.shape[-1])
        bias = getattr(module, "bias", None)
        if bias is not None:
            output = output + bias.to(device="cpu", dtype=output.dtype)
        return output.to(device=flat_input.device, dtype=flat_input.dtype)
    except Exception:
        logger.warning(
            "Could not run ModelOpt NVFP4 CPU sidecar linear fallback for %s",
            state.weight_key,
            exc_info=True,
        )
        return None


def _cached_modelopt_nvfp4_weight(
    module: Any,
    state: _ModelOptNvfp4State,
    *,
    device: Any,
    dtype: Any,
) -> Any | None:
    cache_key = ("modelopt", str(device), str(dtype), state.weight_key)
    cache = getattr(module, _FOUROVERSIX_DEQUANTIZED_WEIGHT_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(module, _FOUROVERSIX_DEQUANTIZED_WEIGHT_CACHE_ATTR, cache)
    cached_weight = cache.get(cache_key)
    if cached_weight is not None:
        return cached_weight
    weight_tensor = _dequantize_modelopt_nvfp4_weight(
        state,
        device=device,
        dtype=dtype,
    )
    if weight_tensor is None:
        state.tensors.clear()
        weight_tensor = _dequantize_modelopt_nvfp4_weight(
            state,
            device=device,
            dtype=dtype,
        )
    if weight_tensor is None and str(device) != "cpu":
        state.tensors.clear()
        cpu_weight = _dequantize_modelopt_nvfp4_weight(
            state,
            device="cpu",
            dtype=dtype,
        )
        if cpu_weight is not None:
            weight_tensor = cpu_weight.to(device=device, dtype=dtype, non_blocking=True)
    if weight_tensor is None:
        weight_tensor = _dequantize_fresh_modelopt_nvfp4_weight(
            state,
            device=device,
            dtype=dtype,
        )
    if weight_tensor is None:
        weight_tensor = _dequantize_modelopt_nvfp4_module_weight(
            module,
            state,
            device=device,
            dtype=dtype,
        )
    if weight_tensor is not None:
        weight_tensor = weight_tensor.detach()
        _maybe_cache_dequantized_weight(cache, cache_key, weight_tensor)
    return weight_tensor


def _uncached_modelopt_nvfp4_weight(
    state: _ModelOptNvfp4State,
    *,
    device: Any,
    dtype: Any,
) -> Any | None:
    state.tensors.clear()
    weight_tensor = _dequantize_modelopt_nvfp4_weight(
        state,
        device=device,
        dtype=dtype,
    )
    if weight_tensor is not None:
        return weight_tensor.detach()
    if str(device) != "cpu":
        state.tensors.clear()
        cpu_weight = _dequantize_modelopt_nvfp4_weight(
            state,
            device="cpu",
            dtype=dtype,
        )
        if cpu_weight is not None:
            try:
                return cpu_weight.to(device=device, dtype=dtype, non_blocking=True).detach()
            except Exception:
                logger.debug(
                    "Could not copy ModelOpt NVFP4 CPU sidecar weight to %s",
                    device,
                    exc_info=True,
                )
    weight_tensor = _dequantize_fresh_modelopt_nvfp4_weight(
        state,
        device=device,
        dtype=dtype,
    )
    return weight_tensor.detach() if weight_tensor is not None else None


def _force_modelopt_nvfp4_sidecar_weight(
    state: _ModelOptNvfp4State,
    *,
    device: Any,
    dtype: Any,
) -> Any | None:
    last_error: Exception | None = None
    stores = [state.store]
    snapshot_path = getattr(state.store, "snapshot_path", None)
    if snapshot_path is not None:
        try:
            stores.append(_SafetensorsKeyStore(Path(snapshot_path)))
        except Exception as exc:
            last_error = exc
            logger.debug("Could not reopen ModelOpt NVFP4 safetensors sidecar", exc_info=True)

    attempt_devices = [device]
    if str(device) != "cpu":
        attempt_devices.append("cpu")
    for store in stores:
        for attempt_device in attempt_devices:
            try:
                weight = store.tensor(state.weight_key)
                scale = store.tensor(state.scale_key)
                global_scale = store.tensor(state.global_scale_key)
                weight_tensor = _dequantize_modelopt_nvfp4_tensors(
                    weight,
                    scale,
                    global_scale,
                    device=attempt_device,
                    dtype=dtype,
                )
            except Exception as exc:
                last_error = exc
                logger.debug("Could not force-read ModelOpt NVFP4 sidecar tensors", exc_info=True)
                continue
            if weight_tensor is None:
                continue
            if str(attempt_device) != str(device):
                try:
                    weight_tensor = weight_tensor.to(
                        device=device,
                        dtype=dtype,
                        non_blocking=True,
                    )
                except Exception as exc:
                    last_error = exc
                    logger.debug(
                        "Could not copy force-read ModelOpt NVFP4 sidecar weight to %s",
                        device,
                        exc_info=True,
                    )
                    continue
            return weight_tensor.detach()
    if last_error is not None:
        logger.debug(
            "ModelOpt NVFP4 sidecar force-read failed for %s: %s",
            state.weight_key,
            last_error,
        )
    return None


def _dequantize_fresh_modelopt_nvfp4_weight(
    state: _ModelOptNvfp4State,
    *,
    device: Any,
    dtype: Any,
) -> Any | None:
    snapshot_path = getattr(state.store, "snapshot_path", None)
    if snapshot_path is None:
        return None
    try:
        fresh_store = _SafetensorsKeyStore(Path(snapshot_path))
    except Exception:
        logger.debug("Could not reopen ModelOpt NVFP4 safetensors sidecar", exc_info=True)
        return None
    fresh_state = _ModelOptNvfp4State(
        store=fresh_store,
        weight_key=state.weight_key,
        scale_key=state.scale_key,
        global_scale_key=state.global_scale_key,
    )
    return _dequantize_modelopt_nvfp4_weight(
        fresh_state,
        device=device,
        dtype=dtype,
    )


def _log_fouroversix_fallback_mismatch_once(
    module: Any,
    reason: str,
    *,
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...] | None,
) -> None:
    marker = "_vllm_colocate_fouroversix_mismatch_logged"
    if getattr(module, marker, False):
        return
    setattr(module, marker, True)
    logger.warning(
        "FourOverSix ModelOpt fallback mismatch reason=%s module=%s input_shape=%s weight_shape=%s",
        reason,
        getattr(module, _MODELOPT_NVFP4_MODULE_NAME_ATTR, "-"),
        input_shape,
        weight_shape,
    )


def _maybe_cache_dequantized_weight(
    cache: dict[Any, Any],
    cache_key: Any,
    weight_tensor: Any,
) -> None:
    global _FOUROVERSIX_DEQUANTIZED_CACHE_BYTES
    max_cache_bytes = _env_int(
        "VLLM_COLOCATE_FOUROVERSIX_DEQUANT_CACHE_MAX_BYTES",
        512 * 1024 * 1024,
    )
    if max_cache_bytes <= 0:
        return
    tensor_bytes = _tensor_nbytes(weight_tensor)
    if tensor_bytes <= 0 or tensor_bytes > max_cache_bytes:
        return
    if _FOUROVERSIX_DEQUANTIZED_CACHE_BYTES + tensor_bytes > max_cache_bytes:
        return
    cache[cache_key] = weight_tensor
    _FOUROVERSIX_DEQUANTIZED_CACHE_BYTES += tensor_bytes


def _tensor_nbytes(tensor: Any) -> int:
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _attach_modelopt_nvfp4_sidecar_tensors(model: Any, model_name: str) -> int:
    snapshot_path = _resolve_model_snapshot_path(model_name)
    if snapshot_path is None:
        return 0
    try:
        store = _SafetensorsKeyStore(snapshot_path)
    except Exception:
        logger.debug("Could not open ModelOpt NVFP4 safetensors sidecar", exc_info=True)
        return 0

    attached = 0
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        setattr(module, _MODELOPT_NVFP4_MODULE_NAME_ATTR, module_name)
        weight_key = f"{module_name}.weight"
        scale_key = f"{module_name}.weight_scale"
        global_scale_key = f"{module_name}.weight_scale_2"
        if (
            weight_key not in store
            or scale_key not in store
            or global_scale_key not in store
        ):
            continue
        setattr(
            module,
            _MODELOPT_NVFP4_STATE_ATTR,
            _ModelOptNvfp4State(
                store=store,
                weight_key=weight_key,
                scale_key=scale_key,
                global_scale_key=global_scale_key,
            ),
        )
        attached += 1
    return attached


def _resolve_model_snapshot_path(model_name: str) -> Path | None:
    path = Path(model_name)
    if path.exists():
        return path
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_name, local_files_only=True))
    except Exception:
        logger.debug(
            "Could not resolve local Hugging Face snapshot for %s",
            model_name,
            exc_info=True,
        )
        return None


def _dequantize_modelopt_nvfp4_weight(
    state: _ModelOptNvfp4State,
    *,
    device: Any,
    dtype: Any,
    block_size: int = 16,
) -> Any | None:
    import torch

    try:
        weight = state.tensor("weight")
        scale = state.tensor("scale")
        global_scale = state.tensor("global_scale")
    except Exception:
        logger.debug("Could not read ModelOpt NVFP4 sidecar tensors", exc_info=True)
        return None

    return _dequantize_modelopt_nvfp4_tensors(
        weight,
        scale,
        global_scale,
        device=device,
        dtype=dtype,
        block_size=block_size,
    )


def _dequantize_modelopt_nvfp4_module_weight(
    module: Any,
    state: _ModelOptNvfp4State,
    *,
    device: Any,
    dtype: Any,
    block_size: int = 16,
) -> Any | None:
    import torch

    weight = _tensor_data(getattr(module, "quantized_weight_values", None), torch)
    scale = _tensor_data(getattr(module, "quantized_weight_scale_factors", None), torch)
    global_scale = _tensor_data(getattr(module, "quantized_weight_amax", None), torch)
    if weight is None or scale is None or global_scale is None:
        weight = _tensor_data(getattr(module, "weight", None), torch)
        if weight is None:
            return None
        try:
            scale = state.tensor("scale")
            global_scale = state.tensor("global_scale")
        except Exception:
            logger.debug("Could not read ModelOpt NVFP4 scale sidecar tensors", exc_info=True)
            return None
    return _dequantize_modelopt_nvfp4_tensors(
        weight,
        scale,
        global_scale,
        device=device,
        dtype=dtype,
        block_size=block_size,
    )


def _tensor_data(value: Any, torch: Any) -> Any | None:
    if isinstance(value, torch.nn.Parameter):
        value = value.data
    return value if isinstance(value, torch.Tensor) else None


def _dequantize_modelopt_nvfp4_tensors(
    weight: Any,
    scale: Any,
    global_scale: Any,
    *,
    device: Any,
    dtype: Any,
    block_size: int = 16,
) -> Any | None:
    import torch

    with torch.no_grad():
        if weight.ndim == 3 and weight.shape[0] == 1:
            weight = weight.squeeze(0)
        if scale.ndim == 3 and scale.shape[0] == 1:
            scale = scale.squeeze(0)
        if weight.ndim != 2 or weight.dtype != torch.uint8:
            return None

        rows, packed_cols = int(weight.shape[0]), int(weight.shape[1])
        logical_cols = packed_cols * 2
        scale_cols = logical_cols // block_size
        if logical_cols % block_size != 0:
            return None

        weight = weight.to(device=device, non_blocking=True)
        scale = scale.to(device=device, non_blocking=True)
        if scale.ndim == 1 and scale.numel() >= rows * scale_cols:
            scale = scale[: rows * scale_cols].reshape(rows, scale_cols)
        if scale.ndim != 2 or scale.shape[0] < rows or scale.shape[1] < scale_cols:
            return None
        scale = scale[:rows, :scale_cols].to(torch.float32)

        global_scale = global_scale.to(device=device, dtype=torch.float32).max()
        values = _break_modelopt_nvfp4_bytes(weight, dtype=torch.float32)
        values = values.reshape(rows, scale_cols, block_size)
        dequantized = values * scale.unsqueeze(-1) * global_scale
        return dequantized.reshape(rows, logical_cols).to(dtype=dtype)


def _break_modelopt_nvfp4_bytes(packed: Any, *, dtype: Any) -> Any:
    import torch

    if packed.dtype != torch.uint8:
        packed = packed.to(torch.uint8)
    flat = packed.flatten()
    low = flat & 0x0F
    high = (flat & 0xF0) >> 4
    nibbles = torch.stack((low, high), dim=1).flatten()
    magnitudes = (nibbles & 0x07).to(torch.long)
    signs = (nibbles & 0x08).to(torch.bool)
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=packed.device,
        dtype=torch.float32,
    )
    values = table[magnitudes]
    values = torch.where(signs, -values, values)
    return values.reshape(packed.shape[0], packed.shape[1] * 2).to(dtype=dtype)


def _install_rotary_embedding_method_compat() -> None:
    try:
        import torch.nn as nn
    except Exception:
        return
    marker = "_vllm_colocate_original_getattr_for_rope"
    if hasattr(nn.Module, marker):
        return
    original_getattr = nn.Module.__getattr__

    def _rope_compat_getattr(self: Any, name: str) -> Any:
        if (
            name == "compute_default_rope_parameters"
            and "RotaryEmbedding" in self.__class__.__name__
        ):
            return lambda config=None, device=None, seq_len=None, layer_type=None: (
                _compute_default_rope_parameters(
                    config or getattr(self, "config", None),
                    device=device
                    or getattr(getattr(self, "inv_freq", None), "device", None),
                    seq_len=seq_len,
                    layer_type=layer_type,
                )
            )
        return original_getattr(self, name)

    setattr(nn.Module, marker, original_getattr)
    nn.Module.__getattr__ = _rope_compat_getattr


def _compute_default_rope_parameters(
    config: Any | None = None,
    device: Any | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple[Any, float]:
    import torch

    rope_parameters = _rope_parameters(config, layer_type)
    base = float(
        rope_parameters.get(
            "rope_theta",
            getattr(config, "rope_theta", getattr(config, "base", 10000.0)),
        )
    )
    partial_rotary_factor = float(
        rope_parameters.get(
            "partial_rotary_factor",
            getattr(config, "partial_rotary_factor", 1.0),
        )
    )
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = int(getattr(config, "hidden_size"))
        attention_heads = int(getattr(config, "num_attention_heads"))
        head_dim = hidden_size // attention_heads
    dim = int(float(head_dim) * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


def _rope_parameters(config: Any | None, layer_type: str | None) -> dict[str, Any]:
    if config is None:
        return {}
    try:
        config.standardize_rope_params()
    except Exception:
        pass
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        if layer_type is not None and isinstance(rope_parameters.get(layer_type), dict):
            return dict(rope_parameters[layer_type])
        return dict(rope_parameters)
    return {}


def _torch_dtype_from_name(name: str) -> Any | None:
    import torch

    normalized = str(name or "").strip().lower()
    if normalized in {"", "auto", "none"}:
        return None
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return aliases.get(normalized)


def _training_quantization_config(config: LoraTrainerConfig) -> Any | None:
    quantization = str(config.quantization or "auto").strip().lower()
    if quantization in {"", "none", "false", "off"}:
        return None
    if quantization == "auto" and "nvfp4" not in config.model_name.lower():
        return None
    if quantization in {"auto", "nvfp4", "fouroversix"}:
        try:
            from transformers import FourOverSixConfig
        except Exception:
            if quantization == "auto":
                logger.warning("FourOverSixConfig unavailable; loading base without NVFP4 training quantization")
                return None
            raise
        if not _fouroversix_runtime_available():
            logger.warning(
                "fouroversix top-level runtime helpers are unavailable; attempting FourOverSixConfig load anyway"
            )
        return FourOverSixConfig(
            dtype="nvfp4",
            keep_master_weights=True,
            weight_scale_2d=True,
            scale_rule="static_6",
            output_dtype="bfloat16",
        )
    if quantization == "fp_quant":
        from transformers import FPQuantConfig

        return FPQuantConfig(
            forward_dtype="nvfp4",
            forward_method="quest",
            backward_dtype="mxfp4",
            store_master_weights=True,
        )
    raise ValueError(f"Unsupported live LoRA quantization mode: {config.quantization}")


def _fouroversix_runtime_available() -> bool:
    try:
        import fouroversix
    except Exception:
        return False
    return all(
        hasattr(fouroversix, name)
        for name in ("QuantizedModule", "quantize_model", "WeightConversions")
    )


def _is_missing_training_dependency_exception(exc: BaseException) -> bool:
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "fouroversix",
            "four oversix",
            "quantizedmodule",
            "quantize_model",
            "weightconversions",
        )
    )


def _use_device_map(config: LoraTrainerConfig) -> bool:
    mode = str(config.parallel_mode or "auto").replace("-", "_").lower()
    if mode in {"device_map", "model_parallel", "auto"}:
        try:
            import torch

            return torch.cuda.is_available() and torch.cuda.device_count() > 1
        except Exception:
            return False
    return False


def _inject_lora_modules(
    model: Any,
    *,
    target_modules: Iterable[str],
    include_expert_lora: bool = False,
    r: int,
    alpha: float,
    dropout: float,
) -> dict[str, Any]:
    import torch

    class _LoRALinear(torch.nn.Module):
        def __init__(self, base: Any, in_features: int, out_features: int):
            super().__init__()
            self.base = base
            for parameter in self.base.parameters(recurse=True):
                parameter.requires_grad = False
            self.r = int(r)
            self.scaling = float(alpha) / float(max(1, r))
            self.dropout = torch.nn.Dropout(float(dropout))
            device, dtype = _module_lora_device_dtype(self.base, torch)
            factory_kwargs = {"device": device, "dtype": dtype}
            self.lora_A = torch.nn.Parameter(
                torch.empty(self.r, in_features, **factory_kwargs)
            )
            self.lora_B = torch.nn.Parameter(
                torch.zeros(out_features, self.r, **factory_kwargs)
            )
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.enabled = True

        def forward(self, input: Any, *args: Any, **kwargs: Any) -> Any:
            result = self.base(input, *args, **kwargs)
            if not self.enabled or self.r <= 0:
                return result
            self._move_lora_to_input_device(input)
            lora_input = self.dropout(input).to(self.lora_A.dtype)
            update = (lora_input @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
            return result + update.to(result.dtype)

        def _move_lora_to_input_device(self, input: Any) -> None:
            input_device = getattr(input, "device", None)
            if input_device is None or self.lora_A.device == input_device:
                return
            self.lora_A.data = self.lora_A.data.to(input_device)
            self.lora_B.data = self.lora_B.data.to(input_device)

    targets = {target.strip() for target in target_modules if target.strip()}
    wrappers: dict[str, Any] = {}
    for module_name, module in list(model.named_modules()):
        if not module_name:
            continue
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf not in targets and module_name not in targets:
            continue
        if not include_expert_lora and _is_expert_module_name(module_name):
            continue
        features = _linear_features(module)
        if features is None:
            continue
        parent, child_name = _parent_module(model, module_name)
        wrapped = _LoRALinear(module, *features)
        setattr(parent, child_name, wrapped)
        wrappers[module_name] = wrapped
    for parameter in model.parameters():
        if parameter.requires_grad and not _parameter_is_lora(parameter, wrappers.values()):
            parameter.requires_grad = False
    return wrappers


def _is_expert_module_name(module_name: str) -> bool:
    normalized = f".{module_name}."
    return ".experts." in normalized


def _module_lora_device_dtype(module: Any, torch: Any) -> tuple[Any, Any]:
    device = None
    dtype = None
    tensors = list(module.parameters(recurse=True)) + list(module.buffers(recurse=True))
    for tensor in tensors:
        if device is None:
            device = tensor.device
        if dtype is None and getattr(tensor.dtype, "is_floating_point", False):
            dtype = tensor.dtype
        if device is not None and dtype is not None:
            break
    return device or torch.device("cpu"), dtype or torch.float32


def _parameter_is_lora(parameter: Any, wrappers: Iterable[Any]) -> bool:
    return any(parameter is wrapper.lora_A or parameter is wrapper.lora_B for wrapper in wrappers)


def _linear_features(module: Any) -> tuple[int, int] | None:
    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    if in_features is not None and out_features is not None:
        return int(in_features), int(out_features)
    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[1]), int(shape[0])
    return None


def _parent_module(model: Any, module_name: str) -> tuple[Any, str]:
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


@contextlib.contextmanager
def _lora_disabled(model: Any):
    wrappers = [module for module in model.modules() if hasattr(module, "lora_A") and hasattr(module, "enabled")]
    previous = [bool(module.enabled) for module in wrappers]
    try:
        for module in wrappers:
            module.enabled = False
        yield
    finally:
        for module, enabled in zip(wrappers, previous):
            module.enabled = enabled


def _first_parameter_device(model: Any, requested_device: str, torch: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        pass
    if str(requested_device).startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda", int(os.getenv("LOCAL_RANK", "0") or 0))
    return torch.device("cpu")


def _planned_steps(config: LoraTrainerConfig, example_count: int) -> int:
    if config.max_steps > 0:
        return config.max_steps
    batches = max(1, math.ceil(max(1, example_count) / max(1, config.batch_size)))
    return max(1, math.ceil(batches * max(0.0, config.train_epochs)))


def _task_for_step(step: int, available_tasks: list[str]) -> str:
    if not available_tasks:
        raise ValueError("No training tasks available")
    return available_tasks[step % len(available_tasks)]


def _sft_batch_for_step(
    examples: list[dict[str, list[int]]],
    step: int,
    accum_index: int,
    batch_size: int,
    pad_token_id: int,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    selected = _select_examples(examples, step, accum_index, batch_size)
    return _collate_token_examples(selected, pad_token_id, torch, device)


def _dpo_batch_for_step(
    examples: list[dict[str, dict[str, list[int]]]],
    step: int,
    accum_index: int,
    batch_size: int,
    pad_token_id: int,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    selected = _select_examples(examples, step, accum_index, batch_size)
    return {
        "chosen": _collate_token_examples(
            [item["chosen"] for item in selected], pad_token_id, torch, device
        ),
        "rejected": _collate_token_examples(
            [item["rejected"] for item in selected], pad_token_id, torch, device
        ),
    }


def _kto_batch_for_step(
    examples: list[dict[str, Any]],
    step: int,
    accum_index: int,
    batch_size: int,
    pad_token_id: int,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    selected = _select_examples(examples, step, accum_index, batch_size)
    batch: dict[str, Any] = {
        "target": _collate_token_examples(
            [item["target"] for item in selected], pad_token_id, torch, device
        ),
        "desirable": torch.tensor(
            [bool(item["desirable"]) for item in selected],
            dtype=torch.bool,
            device=device,
        ),
    }
    kl_items = [item["kl"] for item in selected if item.get("kl") is not None]
    if kl_items:
        batch["kl"] = _collate_token_examples(kl_items, pad_token_id, torch, device)
    return batch


def _select_examples(
    examples: list[Any], step: int, accum_index: int, batch_size: int
) -> list[Any]:
    if not examples:
        raise ValueError("No examples available")
    start = ((step * 9973) + accum_index * batch_size) % len(examples)
    selected = [examples[(start + offset) % len(examples)] for offset in range(batch_size)]
    if len(examples) > batch_size:
        random.Random(step + accum_index).shuffle(selected)
    return selected


def _collate_token_examples(
    examples: list[dict[str, list[int]]], pad_token_id: int, torch: Any, device: Any
) -> dict[str, Any]:
    max_len = max(len(example["input_ids"]) for example in examples)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    for example in examples:
        input_ids = list(example["input_ids"])
        labels = list(example["labels"])
        pad = max_len - len(input_ids)
        input_rows.append(input_ids + [pad_token_id] * pad)
        label_rows.append(labels + [-100] * pad)
        attention_rows.append([1] * len(input_ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "labels": torch.tensor(label_rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.long, device=device),
    }


def _sft_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    loss_vocab_sample_size: int = 0,
    torch: Any,
    F: Any,
) -> Any:
    return _sequence_nll(
        model,
        batch,
        loss_vocab_sample_size=loss_vocab_sample_size,
        torch=torch,
        F=F,
    ).mean()


def _dpo_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    beta: float,
    loss_vocab_sample_size: int = 0,
    torch: Any,
    F: Any,
) -> Any:
    chosen = batch["chosen"]
    rejected = batch["rejected"]
    if loss_vocab_sample_size > 0 and _trainable_parameters_are_output_head_only(model):
        shared_hidden_loss = _dpo_loss_with_shared_hidden(
            model,
            chosen,
            rejected,
            beta=beta,
            loss_vocab_sample_size=loss_vocab_sample_size,
            torch=torch,
            F=F,
        )
        if shared_hidden_loss is not None:
            return shared_hidden_loss
    policy_chosen = _sequence_logprob(
        model,
        chosen,
        loss_vocab_sample_size=loss_vocab_sample_size,
        torch=torch,
        F=F,
    )
    policy_rejected = _sequence_logprob(
        model,
        rejected,
        loss_vocab_sample_size=loss_vocab_sample_size,
        torch=torch,
        F=F,
    )
    with torch.no_grad(), _lora_disabled(model):
        ref_chosen = _sequence_logprob(
            model,
            chosen,
            loss_vocab_sample_size=loss_vocab_sample_size,
            torch=torch,
            F=F,
        )
        ref_rejected = _sequence_logprob(
            model,
            rejected,
            loss_vocab_sample_size=loss_vocab_sample_size,
            torch=torch,
            F=F,
        )
    logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    logits = _zero_nonfinite(logits.float(), torch=torch)
    return -F.logsigmoid(logits).mean()


def _dpo_loss_with_shared_hidden(
    model: Any,
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    *,
    beta: float,
    loss_vocab_sample_size: int,
    torch: Any,
    F: Any,
) -> Any | None:
    chosen_hidden = _causal_lm_hidden_states(model, chosen)
    rejected_hidden = _causal_lm_hidden_states(model, rejected)
    if chosen_hidden is None or rejected_hidden is None:
        return None
    policy_chosen_nll = _sampled_completion_nll_from_hidden(
        model,
        chosen,
        chosen_hidden,
        sample_size=loss_vocab_sample_size,
        torch=torch,
        F=F,
    )
    policy_rejected_nll = _sampled_completion_nll_from_hidden(
        model,
        rejected,
        rejected_hidden,
        sample_size=loss_vocab_sample_size,
        torch=torch,
        F=F,
    )
    if policy_chosen_nll is None or policy_rejected_nll is None:
        return None
    with torch.no_grad(), _lora_disabled(model):
        ref_chosen_nll = _sampled_completion_nll_from_hidden(
            model,
            chosen,
            chosen_hidden.detach(),
            sample_size=loss_vocab_sample_size,
            torch=torch,
            F=F,
        )
        ref_rejected_nll = _sampled_completion_nll_from_hidden(
            model,
            rejected,
            rejected_hidden.detach(),
            sample_size=loss_vocab_sample_size,
            torch=torch,
            F=F,
        )
    if ref_chosen_nll is None or ref_rejected_nll is None:
        return None
    policy_chosen = -policy_chosen_nll
    policy_rejected = -policy_rejected_nll
    ref_chosen = -ref_chosen_nll
    ref_rejected = -ref_rejected_nll
    logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    logits = _zero_nonfinite(logits.float(), torch=torch)
    return -F.logsigmoid(logits).mean()


def _kto_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    beta: float,
    desirable_weight: float,
    undesirable_weight: float,
    loss_vocab_sample_size: int = 0,
    torch: Any,
    F: Any,
) -> Any:
    target = batch["target"]
    policy_logps = _sequence_logprob(
        model,
        target,
        loss_vocab_sample_size=loss_vocab_sample_size,
        torch=torch,
        F=F,
    )
    with torch.no_grad(), _lora_disabled(model):
        ref_logps = _sequence_logprob(
            model,
            target,
            loss_vocab_sample_size=loss_vocab_sample_size,
            torch=torch,
            F=F,
        )
    kl_batch = batch.get("kl")
    if kl_batch is not None:
        with torch.no_grad():
            policy_kl = _sequence_logprob(
                model,
                kl_batch,
                loss_vocab_sample_size=loss_vocab_sample_size,
                torch=torch,
                F=F,
            )
            with _lora_disabled(model):
                ref_kl = _sequence_logprob(
                    model,
                    kl_batch,
                    loss_vocab_sample_size=loss_vocab_sample_size,
                    torch=torch,
                    F=F,
                )
        kl = (policy_kl - ref_kl).float().mean().clamp(min=0.0)
    else:
        kl = policy_logps.detach().new_zeros(())
    logratio = _zero_nonfinite((policy_logps - ref_logps).float(), torch=torch)
    desirable = batch["desirable"].to(logratio.device)
    desirable_losses = float(desirable_weight) * (
        1.0 - torch.sigmoid(beta * (logratio - kl))
    )
    undesirable_losses = float(undesirable_weight) * (
        1.0 - torch.sigmoid(beta * (kl - logratio))
    )
    losses = torch.where(desirable, desirable_losses, undesirable_losses)
    return losses.mean()


def _sequence_logprob(
    model: Any,
    batch: dict[str, Any],
    *,
    loss_vocab_sample_size: int = 0,
    torch: Any,
    F: Any,
) -> Any:
    return -_sequence_nll(
        model,
        batch,
        loss_vocab_sample_size=loss_vocab_sample_size,
        torch=torch,
        F=F,
    )


def _sequence_nll(
    model: Any,
    batch: dict[str, Any],
    *,
    loss_vocab_sample_size: int = 0,
    torch: Any,
    F: Any,
) -> Any:
    if loss_vocab_sample_size > 0:
        sampled_nll = _sampled_completion_nll(
            model,
            batch,
            sample_size=loss_vocab_sample_size,
            torch=torch,
            F=F,
        )
        if sampled_nll is not None:
            return sampled_nll
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    return _completion_nll(outputs.logits, batch["labels"], torch=torch, F=F)


def _sampled_completion_nll(
    model: Any,
    batch: dict[str, Any],
    *,
    sample_size: int,
    torch: Any,
    F: Any,
) -> Any | None:
    hidden = _causal_lm_hidden_states(model, batch)
    return _sampled_completion_nll_from_hidden(
        model,
        batch,
        hidden,
        sample_size=sample_size,
        torch=torch,
        F=F,
    )


def _sampled_completion_nll_from_hidden(
    model: Any,
    batch: dict[str, Any],
    hidden: Any,
    *,
    sample_size: int,
    torch: Any,
    F: Any,
) -> Any | None:
    if hidden is None or getattr(hidden, "ndim", 0) != 3:
        return None
    labels = batch["labels"].to(hidden.device)
    shift_hidden = hidden[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    if not bool(mask.any().detach().cpu()):
        return shift_hidden.new_zeros((shift_hidden.shape[0],), dtype=torch.float32)
    target_ids = shift_labels[mask].to(torch.long)
    vocab_size = _output_vocab_size(model)
    if vocab_size <= 0:
        return None
    candidates = _sampled_vocab_candidates(
        target_ids,
        sample_size=max(1, int(sample_size)),
        vocab_size=vocab_size,
        torch=torch,
        device=shift_hidden.device,
    )
    logits = _selected_lm_head_logits(model, shift_hidden, candidates, torch=torch)
    if logits is None:
        return None
    mask = mask.to(logits.device)
    candidates = candidates.to(logits.device)
    target_ids = target_ids.to(logits.device)
    target_positions = torch.searchsorted(candidates, target_ids)
    valid_logits = logits[mask].float()
    valid_logits = _zero_nonfinite(valid_logits, torch=torch)
    token_nll_values = -F.log_softmax(valid_logits, dim=-1).gather(
        -1, target_positions.unsqueeze(-1)
    ).squeeze(-1)
    token_nll = valid_logits.new_zeros(mask.shape)
    token_nll[mask] = token_nll_values
    denom = mask.sum(dim=1).clamp_min(1).to(token_nll.dtype)
    return token_nll.sum(dim=1) / denom


def _causal_lm_hidden_states(model: Any, batch: dict[str, Any]) -> Any | None:
    import torch

    backbone = None
    for name in ("model", "transformer", "gpt_neox", "base_model"):
        candidate = getattr(model, name, None)
        if candidate is not None and candidate is not model:
            backbone = candidate
            break
    if backbone is None or not callable(getattr(backbone, "forward", None)):
        return None
    grad_context = (
        contextlib.nullcontext()
        if _module_has_trainable_parameters(backbone)
        else torch.no_grad()
    )
    with grad_context:
        try:
            outputs = backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
        except TypeError:
            outputs = backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=True,
            )
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
        hidden = outputs[0]
    return hidden


def _module_has_trainable_parameters(module: Any) -> bool:
    try:
        return any(parameter.requires_grad for parameter in module.parameters())
    except Exception:
        return True


def _zero_nonfinite(tensor: Any, *, torch: Any) -> Any:
    finite = torch.isfinite(tensor)
    if bool(finite.all().detach().cpu()):
        return tensor
    return torch.where(finite, tensor, tensor.detach().new_zeros(tensor.shape))


def _sanitize_nonfinite_gradients(parameters: Iterable[Any], torch: Any) -> int:
    sanitized = 0
    for parameter in parameters:
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        finite = torch.isfinite(grad)
        if bool(finite.all().detach().cpu()):
            continue
        with torch.no_grad():
            grad.copy_(torch.where(finite, grad, torch.zeros_like(grad)))
        sanitized += 1
    return sanitized


def _sanitize_nonfinite_optimizer_state(optimizer: Any, torch: Any) -> int:
    sanitized = 0
    state_values = getattr(optimizer, "state", {}).values()
    for state in state_values:
        if not isinstance(state, dict):
            continue
        for value in state.values():
            if not torch.is_tensor(value):
                continue
            finite = torch.isfinite(value)
            if bool(finite.all().detach().cpu()):
                continue
            with torch.no_grad():
                value.copy_(torch.where(finite, value, torch.zeros_like(value)))
            sanitized += 1
    return sanitized


def _trainable_parameters_are_output_head_only(model: Any) -> bool:
    head_names = {"lm_head", "embed_out", "output", "score"}
    try:
        trainable_names = [
            name
            for name, parameter in model.named_parameters()
            if getattr(parameter, "requires_grad", False)
        ]
    except Exception:
        return False
    if not trainable_names:
        return False
    for name in trainable_names:
        parts = set(name.split("."))
        if not parts.intersection(head_names):
            return False
    return True


def _lora_targets_need_backbone_grad_checkpointing(target_modules: Iterable[str]) -> bool:
    output_head_targets = {"lm_head", "embed_out", "output", "score"}
    targets = {target.rsplit(".", 1)[-1] for target in target_modules}
    return any(target not in output_head_targets for target in targets)


def _sampled_vocab_candidates(
    target_ids: Any,
    *,
    sample_size: int,
    vocab_size: int,
    torch: Any,
    device: Any,
) -> Any:
    unique_targets = torch.unique(target_ids.to(device=device, dtype=torch.long))
    if vocab_size <= sample_size:
        candidates = torch.arange(vocab_size, device=device, dtype=torch.long)
    else:
        stride = max(1, vocab_size // max(1, sample_size))
        negatives = torch.arange(
            0,
            vocab_size,
            stride,
            device=device,
            dtype=torch.long,
        )[:sample_size]
        candidates = torch.cat((unique_targets, negatives), dim=0)
    return torch.unique(candidates.clamp_(0, max(0, vocab_size - 1))).sort().values


def _output_vocab_size(model: Any) -> int:
    head = _output_embeddings(model)
    weight = getattr(head, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    config = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", 0)
    try:
        return int(vocab_size)
    except (TypeError, ValueError):
        return 0


def _output_embeddings(model: Any) -> Any | None:
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        head = getter()
        if head is not None:
            return head
    return getattr(model, "lm_head", None)


def _selected_lm_head_logits(
    model: Any,
    hidden: Any,
    candidate_ids: Any,
    *,
    torch: Any,
) -> Any | None:
    head = _output_embeddings(model)
    if head is None:
        return None
    return _selected_linear_logits(head, hidden, candidate_ids, torch=torch)


def _selected_linear_logits(
    module: Any,
    input: Any,
    candidate_ids: Any,
    *,
    torch: Any,
) -> Any | None:
    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
        base_logits = _selected_linear_logits(
            module.base,
            input,
            candidate_ids,
            torch=torch,
        )
        if base_logits is None:
            full_logits = module(input)
            return full_logits.index_select(-1, candidate_ids.to(full_logits.device))
        if getattr(module, "enabled", True) and int(getattr(module, "r", 0)) > 0:
            move_to_input = getattr(module, "_move_lora_to_input_device", None)
            if callable(move_to_input):
                move_to_input(input)
            lora_input = module.dropout(input).to(module.lora_A.dtype)
            lora_A = module.lora_A.to(device=lora_input.device, dtype=lora_input.dtype)
            lora_B = module.lora_B.index_select(
                0,
                candidate_ids.to(module.lora_B.device),
            ).to(device=lora_input.device, dtype=lora_input.dtype)
            update = (lora_input @ lora_A.t() @ lora_B.t()) * float(module.scaling)
            base_logits = base_logits + update.to(base_logits.dtype)
        return base_logits

    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    selected_weight = weight.index_select(
        0,
        candidate_ids.to(weight.device),
    ).to(device=input.device, dtype=input.dtype)
    flat_input = input.reshape(-1, input.shape[-1])
    output = flat_input @ selected_weight.t()
    bias = getattr(module, "bias", None)
    if bias is not None:
        selected_bias = bias.index_select(
            0,
            candidate_ids.to(bias.device),
        ).to(device=output.device, dtype=output.dtype)
        output = output + selected_bias
    return output.reshape(*input.shape[:-1], output.shape[-1])


def _completion_nll(logits: Any, labels: Any, *, torch: Any, F: Any) -> Any:
    labels = labels.to(logits.device)
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    target = shift_labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    token_nll = -token_log_probs * mask.to(token_log_probs.dtype)
    return token_nll.sum(dim=1) / mask.sum(dim=1).clamp_min(1).to(token_nll.dtype)


def _restore_latest(
    config: LoraTrainerConfig, model: Any, optimizer: Any | None
) -> int:
    latest = _read_json_file(config.checkpoint_dir / "latest.json")
    checkpoint = latest.get("checkpoint_path") or latest.get("checkpoint")
    if not checkpoint:
        return 0
    checkpoint_path = Path(str(checkpoint))
    if not checkpoint_path.is_absolute():
        checkpoint_path = config.checkpoint_dir / checkpoint_path
    try:
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        logger.exception("Failed to restore live LoRA checkpoint %s", checkpoint_path)
        return 0
    state = payload.get("lora_state")
    if isinstance(state, dict):
        _load_lora_state(model, state)
    if optimizer is not None and isinstance(payload.get("optimizer"), dict):
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except Exception:
            logger.warning("Could not restore live LoRA optimizer state", exc_info=True)
    try:
        return int(payload.get("step") or latest.get("step") or 0)
    except (TypeError, ValueError):
        return 0


def _load_lora_state(model: Any, state: dict[str, Any]) -> None:
    wrappers = {
        name: module
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B")
    }
    for name, tensors in state.items():
        module = wrappers.get(name)
        if module is None or not isinstance(tensors, dict):
            continue
        if "lora_A" in tensors and tuple(module.lora_A.shape) == tuple(tensors["lora_A"].shape):
            module.lora_A.data.copy_(tensors["lora_A"].to(module.lora_A.device, module.lora_A.dtype))
        if "lora_B" in tensors and tuple(module.lora_B.shape) == tuple(tensors["lora_B"].shape):
            module.lora_B.data.copy_(tensors["lora_B"].to(module.lora_B.device, module.lora_B.dtype))


def _save(
    config: LoraTrainerConfig,
    model: Any,
    optimizer: Any | None,
    step: int,
) -> None:
    import torch

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = config.checkpoint_dir / f"adapter-step-{step:08d}"
    lora_state = _lora_state_dict(model)
    lora_config = _adapter_config_payload(config, _injected_leaf_targets(lora_state))
    checkpoint_path = config.checkpoint_dir / f"checkpoint-step-{step:08d}.pt"
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    payload = {
        "version": 1,
        "trainer": "lora",
        "step": int(step),
        "model_name": config.model_name,
        "adapter_name": config.adapter_name,
        "saved_at": time.time(),
        "lora_config": lora_config,
        "lora_state": lora_state,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, tmp_path)
    tmp_path.replace(checkpoint_path)
    _export_peft_adapter(config, adapter_dir, lora_state, lora_config)
    latest = {
        "version": 1,
        "trainer": "lora",
        "checkpoint": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "adapter_path": str(adapter_dir),
        "adapter_name": config.adapter_name,
        "step": int(step),
        "saved_at": payload["saved_at"],
        "model_name": config.model_name,
        "lora_config": lora_config,
    }
    latest_path = config.checkpoint_dir / "latest.json"
    latest_tmp = latest_path.with_suffix(latest_path.suffix + ".tmp")
    latest_tmp.write_text(json.dumps(latest, indent=2) + "\n", encoding="utf-8")
    latest_tmp.replace(latest_path)
    _prune_checkpoints(config)


def _lora_state_dict(model: Any) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            state[name] = {
                "lora_A": module.lora_A.detach().cpu(),
                "lora_B": module.lora_B.detach().cpu(),
            }
    return state


def _injected_leaf_targets(lora_state: dict[str, dict[str, Any]]) -> list[str]:
    return sorted({module_name.rsplit(".", 1)[-1] for module_name in lora_state})


def _adapter_config_payload(
    config: LoraTrainerConfig, target_modules: list[str] | None = None
) -> dict[str, Any]:
    return {
        "base_model_name_or_path": config.model_name,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": int(config.lora_r),
        "lora_alpha": float(config.lora_alpha),
        "lora_dropout": float(config.lora_dropout),
        "target_modules": (
            list(target_modules)
            if target_modules
            else list(config.lora_target_modules)
        ),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
    }


def _export_peft_adapter(
    config: LoraTrainerConfig,
    adapter_dir: Path,
    lora_state: dict[str, dict[str, Any]],
    lora_config: dict[str, Any],
) -> None:
    from safetensors.torch import save_file

    tmp_dir = adapter_dir.with_name(adapter_dir.name + ".tmp")
    if tmp_dir.exists():
        _remove_tree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    peft_state: dict[str, Any] = {}
    for module_name, tensors in lora_state.items():
        prefix = f"base_model.model.{module_name}"
        peft_state[f"{prefix}.lora_A.weight"] = tensors["lora_A"].contiguous()
        peft_state[f"{prefix}.lora_B.weight"] = tensors["lora_B"].contiguous()
    save_file(peft_state, str(tmp_dir / "adapter_model.safetensors"))
    (tmp_dir / "adapter_config.json").write_text(
        json.dumps(lora_config, indent=2) + "\n",
        encoding="utf-8",
    )
    (tmp_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "trainer": "lora",
                "adapter_name": config.adapter_name,
                "model_name": config.model_name,
                "exported_at": time.time(),
                "state_format": "peft_lora_safetensors",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if adapter_dir.exists():
        _remove_tree(adapter_dir)
    tmp_dir.replace(adapter_dir)


def _remove_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            child.rmdir()
    path.rmdir()


def _prune_checkpoints(config: LoraTrainerConfig) -> None:
    if config.keep_last_checkpoints <= 0:
        return
    latest = _read_json_file(config.checkpoint_dir / "latest.json")

    def _resolved(value: Any) -> Path | None:
        if not value:
            return None
        path = Path(str(value))
        return path if path.is_absolute() else config.checkpoint_dir / path

    checkpoints = sorted(
        config.checkpoint_dir.glob("checkpoint-step-*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    keep = set(checkpoints[-config.keep_last_checkpoints :])
    latest_checkpoint = _resolved(
        latest.get("checkpoint_path") or latest.get("checkpoint")
    )
    if latest_checkpoint is not None:
        keep.add(latest_checkpoint)
    for checkpoint in checkpoints:
        if checkpoint not in keep:
            try:
                checkpoint.unlink()
            except OSError:
                logger.debug("Could not prune checkpoint %s", checkpoint, exc_info=True)

    adapter_dirs = sorted(
        (
            path
            for path in config.checkpoint_dir.glob("adapter-step-*")
            if path.is_dir() and not path.name.endswith(".tmp")
        ),
        key=lambda path: path.stat().st_mtime,
    )
    keep_adapters = set(adapter_dirs[-config.keep_last_checkpoints :])
    latest_adapter = _resolved(latest.get("adapter_path"))
    if latest_adapter is not None:
        keep_adapters.add(latest_adapter)
    for adapter_dir in adapter_dirs:
        if adapter_dir not in keep_adapters:
            try:
                _remove_tree(adapter_dir)
            except OSError:
                logger.debug(
                    "Could not prune adapter dir %s", adapter_dir, exc_info=True
                )


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _stop_requested(stop_file: Path | None) -> bool:
    if stop_file is None:
        return False
    return stop_file.exists()


def _write_metric(config: LoraTrainerConfig, payload: dict[str, Any]) -> None:
    if config.metrics_path is None:
        return
    if int(os.getenv("RANK", "0") or 0) != 0:
        return
    row = {
        "ts": time.time(),
        "run_id": config.run_id,
        **payload,
    }
    try:
        config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with config.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    except OSError:
        logger.exception("Failed to write live LoRA metric to %s", config.metrics_path)


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("VLLM_COLOCATE_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _configure_torch_matmul_precision(torch: Any) -> None:
    precision = (
        os.getenv("VLLM_COLOCATE_TORCH_FLOAT32_MATMUL_PRECISION") or "high"
    ).strip().lower()
    if precision in {"", "0", "false", "off", "none"}:
        return
    if precision not in {"highest", "high", "medium"}:
        logger.warning(
            "Ignoring invalid VLLM_COLOCATE_TORCH_FLOAT32_MATMUL_PRECISION=%r",
            precision,
        )
        return
    torch.set_float32_matmul_precision(precision)


def _env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw != "":
            return raw
    return default


def _env_bool_any(names: tuple[str, ...], default: bool) -> bool:
    raw = _env_first(*names)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw torch SFT/DPO/KTO LoRA trainer")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--adapter-name",
        default=_env_first(
            "VLLM_COLOCATE_LORA_ADAPTER_NAME",
            default="colocate-lora",
        ),
    )
    parser.add_argument(
        "--task-mix",
        default=_env_first(
            "VLLM_COLOCATE_TRAIN_TASK_MIX",
            default="mixed",
        ),
        choices=("mixed", "sft", "dpo", "kto"),
    )
    parser.add_argument(
        "--device",
        default=_env_first(
            "VLLM_COLOCATE_TRAIN_DEVICE",
            default="cuda",
        ),
    )
    parser.add_argument(
        "--parallel-mode",
        default=_env_first(
            "VLLM_COLOCATE_TRAIN_PARALLEL_MODE",
            default="auto",
        ),
        choices=("auto", "none", "device_map", "model_parallel"),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(
            _env_first(
                "VLLM_COLOCATE_TRAIN_STEPS",
                default="0",
            )
        ),
    )
    parser.add_argument(
        "--train-epochs",
        type=float,
        default=float(
            _env_first(
                "VLLM_COLOCATE_TRAIN_EPOCHS",
                default="1",
            )
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(
            _env_first(
                "VLLM_COLOCATE_BATCH_SIZE",
                default="1",
            )
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=int(
            _env_first(
                "VLLM_COLOCATE_GRADIENT_ACCUMULATION_STEPS",
                default="1",
            )
        ),
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=int(
            _env_first(
                "VLLM_COLOCATE_MAX_SEQ_LEN",
                default="2048",
            )
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(
            _env_first(
                "VLLM_COLOCATE_LR",
                default="0.0002",
            )
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=float(
            _env_first(
                "VLLM_COLOCATE_WEIGHT_DECAY",
                default="0",
            )
        ),
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=float(
            _env_first(
                "VLLM_COLOCATE_MAX_GRAD_NORM",
                default="1",
            )
        ),
    )
    parser.add_argument(
        "--dpo-beta",
        type=float,
        default=float(
            _env_first(
                "VLLM_COLOCATE_DPO_BETA",
                default="0.1",
            )
        ),
    )
    parser.add_argument(
        "--kto-beta",
        type=float,
        default=float(
            _env_first(
                "VLLM_COLOCATE_KTO_BETA",
                default="0.1",
            )
        ),
    )
    parser.add_argument(
        "--kto-desirable-weight",
        type=float,
        default=float(
            _env_first(
                "VLLM_COLOCATE_KTO_DESIRABLE_WEIGHT",
                default="1.0",
            )
        ),
    )
    parser.add_argument(
        "--kto-undesirable-weight",
        type=float,
        default=float(
            _env_first(
                "VLLM_COLOCATE_KTO_UNDESIRABLE_WEIGHT",
                default="1.0",
            )
        ),
    )
    parser.add_argument(
        "--loss-vocab-sample-size",
        type=int,
        default=int(
            _env_first(
                "VLLM_COLOCATE_LOSS_VOCAB_SAMPLE_SIZE",
                default="0",
            )
        ),
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=int(
            _env_first(
                "VLLM_COLOCATE_LORA_R",
                default="16",
            )
        ),
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=float(
            _env_first(
                "VLLM_COLOCATE_LORA_ALPHA",
                default="32",
            )
        ),
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=float(
            _env_first(
                "VLLM_COLOCATE_LORA_DROPOUT",
                default="0.05",
            )
        ),
    )
    parser.add_argument(
        "--lora-target-modules",
        default=_env_first(
            "VLLM_COLOCATE_LORA_TARGET_MODULES",
            default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,qkv_proj,gate_up_proj,wq,wk,wv,wo",
        ),
    )
    parser.add_argument(
        "--include-expert-lora",
        action=argparse.BooleanOptionalAction,
        default=_env_bool_any(
            (
                "VLLM_COLOCATE_INCLUDE_EXPERT_LORA",
            ),
            False,
        ),
    )
    parser.add_argument(
        "--quantization",
        default=_env_first(
            "VLLM_COLOCATE_TRAIN_QUANTIZATION",
            default="auto",
        ),
    )
    parser.add_argument(
        "--torch-dtype",
        default=_env_first(
            "VLLM_COLOCATE_TORCH_DTYPE",
            default="bfloat16",
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=_env_bool_any(
            (
                "VLLM_COLOCATE_TRUST_REMOTE_CODE",
            ),
            True,
        ),
    )
    parser.add_argument(
        "--save-optimizer-state",
        action=argparse.BooleanOptionalAction,
        default=_env_bool_any(
            (
                "VLLM_COLOCATE_SAVE_OPTIMIZER_STATE",
            ),
            True,
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=int(
            _env_first(
                "VLLM_COLOCATE_CHECKPOINT_EVERY",
                default="16",
            )
        ),
    )
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=int(
            _env_first(
                "VLLM_COLOCATE_KEEP_LAST_CHECKPOINTS",
                default="2",
            )
        ),
    )
    parser.add_argument("--stop-file", default=_env_first("VLLM_COLOCATE_STOP_FILE"))
    parser.add_argument("--metrics-path", default=_env_first("VLLM_COLOCATE_METRICS_PATH"))
    parser.add_argument(
        "--run-id",
        default=_env_first(
            "VLLM_COLOCATE_TRAIN_RUN_ID", default=""
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=int(
            _env_first(
                "VLLM_COLOCATE_TRAIN_LOG_EVERY",
                "VLLM_COLOCATE_TRAIN_LOG_EVERY",
                default="1",
            )
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_training(
        LoraTrainerConfig(
            model_name=args.model_name,
            data_path=Path(args.data_path),
            checkpoint_dir=Path(args.checkpoint_dir),
            adapter_name=args.adapter_name,
            task_mix=args.task_mix,
            device=args.device,
            parallel_mode=args.parallel_mode,
            max_steps=args.max_steps,
            train_epochs=args.train_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_seq_len=args.max_seq_len,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            dpo_beta=args.dpo_beta,
            kto_beta=args.kto_beta,
            kto_desirable_weight=args.kto_desirable_weight,
            kto_undesirable_weight=args.kto_undesirable_weight,
            loss_vocab_sample_size=args.loss_vocab_sample_size,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=_split_csv(args.lora_target_modules),
            include_expert_lora=args.include_expert_lora,
            quantization=args.quantization,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
            save_optimizer_state=args.save_optimizer_state,
            checkpoint_every=args.checkpoint_every,
            keep_last_checkpoints=args.keep_last_checkpoints,
            stop_file=Path(args.stop_file) if args.stop_file else None,
            metrics_path=Path(args.metrics_path) if args.metrics_path else None,
            run_id=args.run_id,
            log_every=args.log_every,
        )
    )


if __name__ == "__main__":
    atexit.register(lambda: None)
    raise SystemExit(main())
