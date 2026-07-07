from __future__ import annotations

import argparse
import atexit
import json
import logging
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .jetspec_adapter import (
    JetSpecAdapter,
    JetSpecAdapterConfig,
    load_checkpoint,
    save_checkpoint,
)


_TERMINATION_REQUESTED = False
logger = logging.getLogger("vllm_dflash_jit.trainer")


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
class TrainerConfig:
    model_name: str
    data_path: Path
    checkpoint_dir: Path
    extra_data_paths: tuple[Path, ...] = ()
    suffix_cache_path: Path | None = None
    token_source_mode: str = "text"
    staged_finetune: bool = False
    device: str = "cuda:0"
    parallel_mode: str = "auto"
    max_steps: int = 0
    pretrain_max_steps: int = 0
    finetune_max_steps: int = 0
    batch_size: int = 4
    max_seq_len: int = 1024
    hidden_size: int = 32
    adapter_architecture: str = "pooled_gru"
    num_speculative_tokens: int = 8
    learning_rate: float = 2e-4
    train_epochs: float = 1.0
    pretrain_epochs: float = 1.0
    finetune_epochs: float = 1.0
    pretrain_live_step_ratio: float = 0.0
    max_examples_per_text: int = 4
    example_stride: int = 1
    min_examples: int = 8
    auto_full_pass_each_run: bool = True
    reset_on_data_source_change: bool = True
    checkpoint_every: int = 16
    keep_last_checkpoints: int = 2
    save_optimizer_state: bool = False
    quality_gate_enabled: bool = True
    quality_eval_examples: int = 128
    quality_min_delta: float = 0.0
    promote_intermediate_checkpoints: bool = False
    stop_file: Path | None = None
    metrics_path: Path | None = None
    run_id: str = ""
    log_every: int = 1


@dataclass(frozen=True)
class TrainingRecord:
    text: str
    prompt: str = ""
    completion: str = ""
    prompt_messages: tuple[dict[str, str], ...] = ()
    teacher_distilled: bool = False

    @property
    def has_completion_boundary(self) -> bool:
        return bool(self.completion)


@dataclass(frozen=True)
class TokenSequenceRecord:
    tokens: list[int]
    min_next_index: int = 1


@dataclass(frozen=True)
class TrainingStage:
    name: str
    data_source: str
    examples: list[tuple[list[int], list[int]]]
    planned_steps: int
    promote_checkpoints: bool


@dataclass(frozen=True)
class ParallelContext:
    mode: str
    distributed: bool
    world_size: int
    rank: int
    local_rank: int
    primary: bool
    device: Any
    local_batch_size: int


def run_training(config: TrainerConfig) -> int:
    import torch
    from transformers import AutoTokenizer

    _configure_logging()
    _install_signal_handlers()
    parallel = _setup_parallel_context(torch, config)
    atexit.register(_destroy_parallel_context, torch, parallel)
    run_started = time.time()

    logger.info(
        "JetSpec trainer starting run_id=%s model=%s data_path=%s checkpoint_dir=%s requested_max_steps=%d parallel_mode=%s world_size=%d rank=%d local_rank=%d local_batch_size=%d",
        config.run_id or "-",
        config.model_name,
        config.data_path,
        config.checkpoint_dir,
        config.max_steps,
        parallel.mode,
        parallel.world_size,
        parallel.rank,
        parallel.local_rank,
        parallel.local_batch_size,
    )
    _write_metric(
        config,
        {
            "event": "run_start",
            "model_name": config.model_name,
            "data_path": str(config.data_path),
            "extra_data_paths": [str(path) for path in config.extra_data_paths],
            "checkpoint_dir": str(config.checkpoint_dir),
            "suffix_cache_path": str(config.suffix_cache_path or ""),
            "token_source_mode": config.token_source_mode,
            "staged_finetune": config.staged_finetune,
            "parallel_mode": parallel.mode,
            "distributed": parallel.distributed,
            "world_size": parallel.world_size,
            "rank": parallel.rank,
            "local_rank": parallel.local_rank,
            "local_batch_size": parallel.local_batch_size,
            "max_steps": config.max_steps,
            "pretrain_max_steps": config.pretrain_max_steps,
            "finetune_max_steps": config.finetune_max_steps,
            "requested_max_steps": config.max_steps,
            "batch_size": config.batch_size,
            "max_seq_len": config.max_seq_len,
            "hidden_size": config.hidden_size,
            "adapter_architecture": config.adapter_architecture,
            "num_speculative_tokens": config.num_speculative_tokens,
            "learning_rate": config.learning_rate,
            "train_epochs": config.train_epochs,
            "pretrain_epochs": config.pretrain_epochs,
            "finetune_epochs": config.finetune_epochs,
            "pretrain_live_step_ratio": config.pretrain_live_step_ratio,
            "max_examples_per_text": config.max_examples_per_text,
            "example_stride": config.example_stride,
            "min_examples": config.min_examples,
            "auto_full_pass_each_run": config.auto_full_pass_each_run,
            "reset_on_data_source_change": config.reset_on_data_source_change,
            "quality_gate_enabled": config.quality_gate_enabled,
            "quality_eval_examples": config.quality_eval_examples,
            "quality_min_delta": config.quality_min_delta,
            "promote_intermediate_checkpoints": config.promote_intermediate_checkpoints,
        },
    )

    if _stop_requested(config.stop_file):
        logger.info(
            "JetSpec trainer stopped before loading data because stop was requested"
        )
        _write_metric(config, {"event": "stopped_before_start"})
        return 0

    primary_records = _load_training_records(config.data_path)
    extra_records = _load_training_records_from_paths(config.extra_data_paths)
    records = (
        primary_records + extra_records
        if not config.staged_finetune
        else primary_records
    )
    texts = [record.text for record in primary_records + extra_records]
    token_sequence_records = _load_token_sequence_records(config.suffix_cache_path)
    token_source_mode = _normalize_token_source_mode(config.token_source_mode)
    use_text_records = token_source_mode in {"text", "mixed"} or (
        token_source_mode == "auto" and bool(primary_records or extra_records)
    )
    use_token_sequences = token_source_mode in {"tokens", "mixed"} or (
        token_source_mode == "auto" and not (primary_records or extra_records)
    )
    data_source = _training_data_source(
        staged_finetune=config.staged_finetune,
        use_text_records=use_text_records,
        use_token_sequences=use_token_sequences,
        primary_records=primary_records,
        extra_records=extra_records,
        token_sequence_records=token_sequence_records,
    )
    source_examples = _source_example_count(
        staged_finetune=config.staged_finetune,
        use_text_records=use_text_records,
        use_token_sequences=use_token_sequences,
        primary_records=primary_records,
        extra_records=extra_records,
        token_sequence_records=token_sequence_records,
    )
    logger.info(
        "JetSpec trainer loaded primary_training_records=%d extra_training_records=%d completion_boundary_records=%d token_sequence_records=%d data_source=%s token_source_mode=%s staged_finetune=%s",
        len(primary_records),
        len(extra_records),
        sum(
            1
            for record in [*primary_records, *extra_records]
            if record.has_completion_boundary
        ),
        len(token_sequence_records),
        data_source,
        token_source_mode,
        config.staged_finetune,
    )
    if source_examples < config.min_examples:
        logger.info(
            "JetSpec trainer skipping run: only %d source example(s), need at least %d",
            source_examples,
            config.min_examples,
        )
        _write_metric(
            config,
            {
                "event": "not_enough_examples",
                "examples": source_examples,
                "data_source": data_source,
                "token_source_mode": token_source_mode,
            },
        )
        return 0

    logger.info("JetSpec trainer loading tokenizer for %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    pad_token_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)
    examples: list[tuple[list[int], list[int]]] = []
    pretrain_examples: list[tuple[list[int], list[int]]] = []
    finetune_examples: list[tuple[list[int], list[int]]] = []
    filtered_token_sequence_records = token_sequence_records
    if config.staged_finetune:
        if use_text_records:
            pretrain_examples.extend(
                _build_examples_from_training_records(
                    tokenizer=tokenizer,
                    records=primary_records,
                    max_seq_len=config.max_seq_len,
                    num_speculative_tokens=config.num_speculative_tokens,
                    label_pad_token_id=pad_token_id,
                    allow_partial_labels=True,
                    max_examples_per_text=config.max_examples_per_text,
                    example_stride=config.example_stride,
                )
            )
            finetune_examples.extend(
                _build_examples_from_training_records(
                    tokenizer=tokenizer,
                    records=extra_records,
                    max_seq_len=config.max_seq_len,
                    num_speculative_tokens=config.num_speculative_tokens,
                    label_pad_token_id=pad_token_id,
                    allow_partial_labels=True,
                    max_examples_per_text=config.max_examples_per_text,
                    example_stride=config.example_stride,
                )
            )
        if use_token_sequences:
            filtered_token_sequence_records = _filter_token_sequence_records(
                token_sequence_records,
                vocab_size=len(tokenizer),
            )
            finetune_examples.extend(
                _build_examples_from_token_sequence_records(
                    token_sequences=filtered_token_sequence_records,
                    max_seq_len=config.max_seq_len,
                    num_speculative_tokens=config.num_speculative_tokens,
                    label_pad_token_id=pad_token_id,
                    allow_partial_labels=True,
                    max_examples_per_sequence=config.max_examples_per_text,
                    example_stride=config.example_stride,
                )
            )
        examples = pretrain_examples + finetune_examples
    else:
        if use_text_records:
            examples.extend(
                _build_examples_from_training_records(
                    tokenizer=tokenizer,
                    records=primary_records + extra_records,
                    max_seq_len=config.max_seq_len,
                    num_speculative_tokens=config.num_speculative_tokens,
                    label_pad_token_id=pad_token_id,
                    allow_partial_labels=True,
                    max_examples_per_text=config.max_examples_per_text,
                    example_stride=config.example_stride,
                )
            )
        if use_token_sequences:
            filtered_token_sequence_records = _filter_token_sequence_records(
                token_sequence_records,
                vocab_size=len(tokenizer),
            )
            examples.extend(
                _build_examples_from_token_sequence_records(
                    token_sequences=filtered_token_sequence_records,
                    max_seq_len=config.max_seq_len,
                    num_speculative_tokens=config.num_speculative_tokens,
                    label_pad_token_id=pad_token_id,
                    allow_partial_labels=True,
                    max_examples_per_sequence=config.max_examples_per_text,
                    example_stride=config.example_stride,
                )
            )
    training_recipe = _training_recipe_id(data_source, config)
    logger.info(
        "JetSpec trainer built %d tokenized training example(s) from %d source row(s) pretrain_examples=%d finetune_examples=%d data_source=%s training_recipe=%s",
        len(examples),
        source_examples,
        len(pretrain_examples),
        len(finetune_examples),
        data_source,
        training_recipe,
    )
    if len(examples) < config.min_examples:
        logger.info(
            "JetSpec trainer skipping run: only %d tokenized example(s), need at least %d",
            len(examples),
            config.min_examples,
        )
        _write_metric(
            config,
            {"event": "not_enough_tokenized_examples", "examples": len(examples)},
        )
        return 0
    stages, train_examples, quality_examples = _build_training_stages(
        config,
        examples=examples,
        pretrain_examples=pretrain_examples,
        finetune_examples=finetune_examples,
        data_source=data_source,
    )
    planned_max_steps = sum(stage.planned_steps for stage in stages)
    train_steps = planned_max_steps
    logger.info(
        "JetSpec trainer planned %d step(s) across %d stage(s) for %d train example(s), %d quality holdout example(s), batch_size=%d, train_epochs=%.3f pretrain_epochs=%.3f finetune_epochs=%.3f pretrain_live_step_ratio=%.3f",
        planned_max_steps,
        len(stages),
        len(train_examples),
        len(quality_examples),
        config.batch_size,
        config.train_epochs,
        config.pretrain_epochs,
        config.finetune_epochs,
        config.pretrain_live_step_ratio,
    )

    device = parallel.device
    logger.info(
        "JetSpec trainer using device=%s parallel_mode=%s world_size=%d local_batch_size=%d",
        device,
        parallel.mode,
        parallel.world_size,
        parallel.local_batch_size,
    )
    adapter_config = JetSpecAdapterConfig(
        vocab_size=len(tokenizer),
        hidden_size=config.hidden_size,
        num_speculative_tokens=config.num_speculative_tokens,
        pad_token_id=pad_token_id,
        architecture=config.adapter_architecture,
    )
    model = JetSpecAdapter(adapter_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer_for_restore = optimizer if config.save_optimizer_state else None
    prior_checkpoint_quality = None
    quality_eval_examples = quality_examples or train_examples
    if parallel.primary and config.quality_gate_enabled:
        prior_checkpoint_quality = _evaluate_latest_checkpoint_quality(
            config.checkpoint_dir,
            quality_eval_examples,
            pad_token_id=pad_token_id,
            device=device,
            batch_size=config.batch_size,
            max_examples=config.quality_eval_examples,
            expected_data_source=data_source,
            expected_training_recipe=training_recipe,
            reset_on_mismatch=config.reset_on_data_source_change,
        )
    _distributed_barrier(torch, parallel)
    step = _restore_latest(
        config.checkpoint_dir,
        model,
        optimizer_for_restore,
        expected_data_source=data_source,
        expected_training_recipe=training_recipe,
        reset_on_mismatch=config.reset_on_data_source_change,
    )
    if step > 0:
        logger.info("JetSpec trainer restored checkpoint at global_step=%d", step)
    else:
        step_floor = _max_checkpoint_step(config.checkpoint_dir)
        if step_floor > 0:
            step = step_floor
            logger.info(
                "JetSpec trainer starting fresh recipe above existing checkpoint step floor=%d",
                step_floor,
            )
    if config.staged_finetune:
        train_steps = planned_max_steps
    else:
        train_steps = _planned_train_steps(config, planned_max_steps, step)
        if stages:
            stages = [
                TrainingStage(
                    name=stages[0].name,
                    data_source=stages[0].data_source,
                    examples=stages[0].examples,
                    planned_steps=train_steps,
                    promote_checkpoints=stages[0].promote_checkpoints,
                )
            ]
    if config.max_steps <= 0 and step > 0 and not config.staged_finetune:
        logger.info(
            "JetSpec trainer will run %d auto step(s) for current traffic target=%d from restored global_step=%d full_pass_each_run=%s",
            train_steps,
            planned_max_steps,
            step,
            config.auto_full_pass_each_run,
        )
    _write_metric(
        config,
        {
            "event": "ready",
            "traffic_examples": len(texts),
            "token_sequence_examples": len(token_sequence_records),
            "data_source": data_source,
            "token_source_mode": token_source_mode,
            "staged_finetune": config.staged_finetune,
            "training_recipe": training_recipe,
            "tokenized_examples": len(examples),
            "pretrain_examples": len(pretrain_examples),
            "finetune_examples": len(finetune_examples),
            "train_examples": len(train_examples),
            "quality_examples": len(quality_examples),
            "stages": [
                {
                    "name": stage.name,
                    "data_source": stage.data_source,
                    "examples": len(stage.examples),
                    "planned_steps": stage.planned_steps,
                    "promote_checkpoints": stage.promote_checkpoints,
                }
                for stage in stages
            ],
            "device": str(device),
            "parallel_mode": parallel.mode,
            "distributed": parallel.distributed,
            "world_size": parallel.world_size,
            "rank": parallel.rank,
            "local_rank": parallel.local_rank,
            "local_batch_size": parallel.local_batch_size,
            "restored_step": step,
            "vocab_size": adapter_config.vocab_size,
            "planned_max_steps": planned_max_steps,
            "train_steps": train_steps,
            "auto_full_pass_each_run": config.auto_full_pass_each_run,
            "reset_on_data_source_change": config.reset_on_data_source_change,
            "example_stride": config.example_stride,
            "pretrain_live_step_ratio": config.pretrain_live_step_ratio,
        },
    )

    model.to(device)
    train_model = _wrap_parallel_model(torch, model, parallel)
    baseline_quality = None
    promoted_baseline_quality = baseline_quality
    if (
        parallel.primary
        and config.quality_gate_enabled
        and prior_checkpoint_quality is not None
    ):
        baseline_quality = prior_checkpoint_quality
        promoted_baseline_quality = baseline_quality
        logger.info(
            "JetSpec trainer baseline quality from latest checkpoint %s exact_prefix_mean=%.3f top1=%.3f examples=%d",
            baseline_quality.get("checkpoint_path", ""),
            baseline_quality["exact_prefix_mean"],
            baseline_quality["top1_accuracy"],
            baseline_quality["examples"],
        )
        _write_metric(config, {"event": "baseline_quality", **baseline_quality})
    elif parallel.primary and config.quality_gate_enabled and step > 0:
        baseline_quality = _evaluate_draft_quality(
            model,
            quality_eval_examples,
            pad_token_id=pad_token_id,
            device=device,
            batch_size=config.batch_size,
            max_examples=config.quality_eval_examples,
        )
        promoted_baseline_quality = baseline_quality
        logger.info(
            "JetSpec trainer baseline quality before run exact_prefix_mean=%.3f top1=%.3f examples=%d",
            baseline_quality["exact_prefix_mean"],
            baseline_quality["top1_accuracy"],
            baseline_quality["examples"],
        )
        _write_metric(config, {"event": "baseline_quality", **baseline_quality})
    _distributed_barrier(torch, parallel)
    train_model.train()
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed_steps = 0
    last_loss = None
    for stage in stages:
        if stage.planned_steps <= 0 or not stage.examples:
            continue
        logger.info(
            "JetSpec trainer starting stage=%s data_source=%s examples=%d planned_steps=%d promote_checkpoints=%s",
            stage.name,
            stage.data_source,
            len(stage.examples),
            stage.planned_steps,
            stage.promote_checkpoints,
        )
        _write_metric(
            config,
            {
                "event": "stage_start",
                "stage": stage.name,
                "stage_data_source": stage.data_source,
                "stage_examples": len(stage.examples),
                "stage_steps": stage.planned_steps,
                "promote_checkpoints": stage.promote_checkpoints,
            },
        )
        stage_examples = _stage_examples_for_rank(stage.examples, parallel)
        stage_completed_steps = 0
        for stage_step, batch in enumerate(
            _iter_training_batches(
                stage_examples,
                batch_size=parallel.local_batch_size,
                max_steps=stage.planned_steps,
            ),
            start=1,
        ):
            if _stop_requested(config.stop_file):
                logger.info(
                    "JetSpec trainer stopping early at stage=%s stage_step=%d/%d global_step=%d",
                    stage.name,
                    stage_step,
                    stage.planned_steps,
                    step,
                )
                break
            input_ids, labels = _collate(
                batch, pad_token_id=pad_token_id, device=device
            )
            try:
                logits = _model_forward(train_model, input_ids, labels)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, adapter_config.vocab_size),
                    labels.reshape(-1),
                    ignore_index=pad_token_id,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            except torch.OutOfMemoryError as exc:
                _handle_cuda_oom(
                    config,
                    torch,
                    exc,
                    parallel=parallel,
                    stage=stage.name,
                    stage_step=stage_step,
                    global_step=step,
                    batch_size=len(batch),
                )
                if parallel.distributed:
                    os._exit(75)
                return 75
            step += 1
            completed_steps += 1
            stage_completed_steps += 1
            last_loss = float(loss.detach().cpu())
            if config.checkpoint_every > 0 and step % config.checkpoint_every == 0:
                checkpoint_quality = None
                checkpoint_promoted = (
                    stage.promote_checkpoints and not config.quality_gate_enabled
                )
                if (
                    parallel.primary
                    and config.quality_gate_enabled
                    and config.promote_intermediate_checkpoints
                    and stage.promote_checkpoints
                ):
                    checkpoint_quality = _evaluate_draft_quality(
                        model,
                        quality_eval_examples,
                        pad_token_id=pad_token_id,
                        device=device,
                        batch_size=config.batch_size,
                        max_examples=config.quality_eval_examples,
                    )
                    checkpoint_promoted = _should_promote_quality(
                        promoted_baseline_quality,
                        checkpoint_quality,
                        min_delta=config.quality_min_delta,
                    )
                    logger.info(
                        "JetSpec trainer intermediate quality stage=%s global_step=%d exact_prefix_mean=%.3f top1=%.3f examples=%d promoted=%s",
                        stage.name,
                        step,
                        checkpoint_quality["exact_prefix_mean"],
                        checkpoint_quality["top1_accuracy"],
                        checkpoint_quality["examples"],
                        checkpoint_promoted,
                    )
                    _write_metric(
                        config,
                        {
                            "event": "intermediate_quality",
                            "stage": stage.name,
                            "step": step,
                            "promoted": checkpoint_promoted,
                            "baseline_quality": promoted_baseline_quality,
                            **checkpoint_quality,
                        },
                    )
                    if checkpoint_promoted:
                        promoted_baseline_quality = checkpoint_quality
                if parallel.primary:
                    _save(
                        config,
                        model,
                        optimizer,
                        step,
                        {
                            "loss": last_loss,
                            "data_source": data_source,
                            "stage": stage.name,
                            "stage_data_source": stage.data_source,
                            "training_recipe": training_recipe,
                            "quality": checkpoint_quality,
                            "intermediate": True,
                            "parallel_mode": parallel.mode,
                            "world_size": parallel.world_size,
                        },
                        promote_latest=checkpoint_promoted,
                    )
                    logger.info(
                        "JetSpec trainer checkpoint saved stage=%s global_step=%d promoted=%s",
                        stage.name,
                        step,
                        checkpoint_promoted,
                    )
                _distributed_barrier(torch, parallel)
            if config.log_every > 0 and (
                stage_step == 1
                or stage_step == stage.planned_steps
                or stage_step % config.log_every == 0
            ):
                logger.info(
                    "JetSpec trainer progress run_id=%s stage=%s stage_step=%d/%d completed_steps=%d/%d global_step=%d loss=%.6f",
                    config.run_id or "-",
                    stage.name,
                    stage_step,
                    stage.planned_steps,
                    completed_steps,
                    train_steps,
                    step,
                    last_loss,
                )
            _write_metric(
                config,
                {
                    "event": "step",
                    "stage": stage.name,
                    "step": step,
                    "local_step": completed_steps,
                    "stage_step": stage_step,
                    "stage_steps": stage.planned_steps,
                    "max_steps": train_steps,
                    "planned_max_steps": planned_max_steps,
                    "requested_max_steps": config.max_steps,
                    "progress": completed_steps / train_steps
                    if train_steps > 0
                    else 1.0,
                    "stage_progress": stage_step / stage.planned_steps
                    if stage.planned_steps > 0
                    else 1.0,
                    "loss": last_loss,
                    "elapsed_seconds": time.time() - run_started,
                },
            )
        _write_metric(
            config,
            {
                "event": "stage_complete",
                "stage": stage.name,
                "completed_steps": stage_completed_steps,
                "stage_steps": stage.planned_steps,
                "loss": last_loss,
                "elapsed_seconds": time.time() - run_started,
            },
        )
        if _stop_requested(config.stop_file):
            break

    stopped = _stop_requested(config.stop_file)
    final_quality = None
    promoted = True
    if parallel.primary and config.quality_gate_enabled:
        final_quality = _evaluate_draft_quality(
            model,
            quality_eval_examples,
            pad_token_id=pad_token_id,
            device=device,
            batch_size=config.batch_size,
            max_examples=config.quality_eval_examples,
        )
        promoted = _should_promote_quality(
            promoted_baseline_quality,
            final_quality,
            min_delta=config.quality_min_delta,
        )
        if completed_steps <= 0:
            promoted = False
        logger.info(
            "JetSpec trainer final quality exact_prefix_mean=%.3f top1=%.3f examples=%d promoted=%s",
            final_quality["exact_prefix_mean"],
            final_quality["top1_accuracy"],
            final_quality["examples"],
            promoted,
        )
        _write_metric(
            config,
            {
                "event": "final_quality",
                "promoted": promoted,
                "baseline_quality": promoted_baseline_quality,
                **final_quality,
            },
        )
    if parallel.primary:
        _save(
            config,
            model,
            optimizer,
            step,
            {
                "final": True,
                "loss": last_loss,
                "data_source": data_source,
                "training_recipe": training_recipe,
                "quality": final_quality,
                "promoted": promoted,
                "parallel_mode": parallel.mode,
                "world_size": parallel.world_size,
            },
            promote_latest=promoted,
        )
        logger.info(
            "JetSpec trainer finished run_id=%s completed_steps=%d/%d global_step=%d stopped=%s elapsed=%.1fs",
            config.run_id or "-",
            completed_steps,
            train_steps,
            step,
            stopped,
            time.time() - run_started,
        )
        _write_metric(
            config,
            {
                "event": "run_complete",
                "step": step,
                "completed_steps": completed_steps,
                "max_steps": train_steps,
                "planned_max_steps": planned_max_steps,
                "requested_max_steps": config.max_steps,
                "loss": last_loss,
                "stopped": stopped,
                "promoted": promoted,
                "baseline_quality": promoted_baseline_quality,
                "final_quality": final_quality,
                "parallel_mode": parallel.mode,
                "world_size": parallel.world_size,
                "elapsed_seconds": time.time() - run_started,
            },
        )
    _distributed_barrier(torch, parallel)
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def _configure_logging() -> None:
    if logger.handlers:
        return
    configured_level = getattr(
        logging,
        (
            _env_first(
                "VLLM_JETSPEC_TRAIN_LOG_LEVEL",
                "VLLM_DFLASH_TRAIN_LOG_LEVEL",
                default="INFO",
            )
            or "INFO"
        ).upper(),
        logging.INFO,
    )
    if not _is_primary_rank() and not _env_bool_any(
        ("VLLM_JETSPEC_LOG_ALL_TRAIN_RANKS", "VLLM_DFLASH_LOG_ALL_TRAIN_RANKS"),
        False,
    ):
        configured_level = max(configured_level, logging.WARNING)
    logging.basicConfig(
        level=configured_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def _setup_parallel_context(torch: Any, config: TrainerConfig) -> ParallelContext:
    mode = _normalize_parallel_mode(config.parallel_mode)
    world_size = _env_rank_int("WORLD_SIZE", 1)
    rank = _env_rank_int("RANK", 0)
    local_rank = _env_rank_int("LOCAL_RANK", 0)
    distributed = bool(torch.cuda.is_available() and world_size > 1 and mode != "none")
    if distributed:
        if mode == "auto":
            mode = "ddp"
        if mode != "ddp":
            logger.warning(
                "Unsupported JetSpec trainer parallel_mode=%s; falling back to DDP",
                mode,
            )
            mode = "ddp"
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        world_size = 1
        rank = 0
        local_rank = 0
        if not torch.cuda.is_available():
            device = torch.device("cpu")
            mode = "none"
        else:
            device = torch.device(config.device)
            if str(device) == "cuda":
                device = torch.device("cuda:0")
            if mode == "auto":
                mode = "none"
    local_batch_size = _local_batch_size(config.batch_size, world_size)
    return ParallelContext(
        mode=mode,
        distributed=distributed,
        world_size=max(1, world_size),
        rank=max(0, rank),
        local_rank=max(0, local_rank),
        primary=(rank == 0),
        device=device,
        local_batch_size=local_batch_size,
    )


def _normalize_parallel_mode(value: str | None) -> str:
    normalized = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "off": "none",
        "false": "none",
        "single": "none",
        "single_gpu": "none",
        "data": "ddp",
        "data_parallel": "ddp",
        "distributed": "ddp",
        "distributed_data_parallel": "ddp",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"auto", "none", "ddp"}:
        return "auto"
    return normalized


def _env_rank_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _is_primary_rank() -> bool:
    return _env_rank_int("RANK", 0) == 0


def _local_batch_size(global_batch_size: int, world_size: int) -> int:
    return max(1, math.ceil(max(1, int(global_batch_size)) / max(1, int(world_size))))


def _wrap_parallel_model(torch: Any, model: Any, parallel: ParallelContext) -> Any:
    if not parallel.distributed:
        return model
    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(
        model,
        device_ids=[parallel.local_rank],
        output_device=parallel.local_rank,
        find_unused_parameters=False,
    )


def _stage_examples_for_rank(
    examples: list[tuple[list[int], list[int]]], parallel: ParallelContext
) -> list[tuple[list[int], list[int]]]:
    if not parallel.distributed:
        return examples
    shard = examples[parallel.rank :: parallel.world_size]
    return shard if shard else list(examples)


def _distributed_barrier(torch: Any, parallel: ParallelContext) -> None:
    if parallel.distributed and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _destroy_parallel_context(torch: Any, parallel: ParallelContext) -> None:
    if not parallel.distributed:
        return
    distributed = getattr(torch, "distributed", None)
    if distributed is None:
        return
    is_initialized = getattr(distributed, "is_initialized", None)
    destroy_process_group = getattr(distributed, "destroy_process_group", None)
    if callable(is_initialized) and callable(destroy_process_group) and is_initialized():
        destroy_process_group()


def _handle_cuda_oom(
    config: TrainerConfig,
    torch: Any,
    exc: BaseException,
    *,
    parallel: ParallelContext,
    stage: str,
    stage_step: int,
    global_step: int,
    batch_size: int,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.error(
        "JetSpec trainer CUDA OOM run_id=%s stage=%s stage_step=%d global_step=%d "
        "device=%s rank=%d/%d batch_size=%d local_batch_size=%d parallel_mode=%s: %s",
        config.run_id or "-",
        stage,
        stage_step,
        global_step,
        parallel.device,
        parallel.rank,
        parallel.world_size,
        batch_size,
        parallel.local_batch_size,
        parallel.mode,
        exc,
    )
    _write_metric(
        config,
        {
            "event": "cuda_oom",
            "stage": stage,
            "stage_step": stage_step,
            "step": global_step,
            "device": str(parallel.device),
            "rank": parallel.rank,
            "world_size": parallel.world_size,
            "batch_size": batch_size,
            "local_batch_size": parallel.local_batch_size,
            "parallel_mode": parallel.mode,
            "error": str(exc),
        },
    )


def _load_texts(path: Path) -> list[str]:
    return [record.text for record in _load_training_records(path)]


def _load_training_records_from_paths(paths: tuple[Path, ...]) -> list[TrainingRecord]:
    records: list[TrainingRecord] = []
    for path in paths:
        records.extend(_load_training_records(path))
    return records


def _load_training_records(path: Path) -> list[TrainingRecord]:
    if not path.exists():
        return []
    records: list[TrainingRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            record = _row_to_training_record(row)
            if record is not None and record.text:
                records.append(record)
    return records


def _row_to_training_record(row: dict[str, Any]) -> TrainingRecord | None:
    prompt = str(row.get("prompt") or "")
    completion = str(row.get("completion") or "")
    teacher_distilled = _row_has_teacher_completion(row)
    messages = row.get("messages")
    if isinstance(messages, list):
        normalized_messages = _normalize_messages(messages)
        prompt_messages, final_completion = _split_normalized_messages_for_training(
            normalized_messages
        )
        if completion:
            if not prompt_messages:
                prompt_messages = normalized_messages
            return TrainingRecord(
                text=prompt + completion,
                prompt=prompt,
                completion=completion,
                prompt_messages=tuple(prompt_messages),
                teacher_distilled=teacher_distilled,
            )
        if final_completion:
            prompt_text = _flatten_prompt_messages(prompt_messages)
            return TrainingRecord(
                text=prompt_text + final_completion,
                prompt=prompt_text,
                completion=final_completion,
                prompt_messages=tuple(prompt_messages),
                teacher_distilled=teacher_distilled,
            )
        text = _flatten_normalized_messages(normalized_messages)
        if text:
            return TrainingRecord(
                text=text,
                prompt_messages=tuple(normalized_messages),
                teacher_distilled=teacher_distilled,
            )

    if completion:
        return TrainingRecord(
            text=prompt + completion,
            prompt=prompt,
            completion=completion,
            teacher_distilled=teacher_distilled,
        )

    text = str(row.get("text") or "")
    if text:
        return TrainingRecord(text=text.strip(), teacher_distilled=teacher_distilled)
    return None


def _row_has_teacher_completion(row: dict[str, Any]) -> bool:
    if row.get("teacher_completion") or row.get("teacher_model"):
        return True
    distillation = row.get("distillation")
    if isinstance(distillation, dict):
        return bool(distillation.get("teacher_model") or distillation.get("source"))
    return False


def _row_to_training_text(row: dict[str, Any]) -> str:
    record = _row_to_training_record(row)
    return record.text if record is not None else ""


def _split_messages_for_training(
    messages: list[Any],
) -> tuple[list[dict[str, str]], str]:
    return _split_normalized_messages_for_training(_normalize_messages(messages))


def _split_normalized_messages_for_training(
    normalized: list[dict[str, str]],
) -> tuple[list[dict[str, str]], str]:
    for index in range(len(normalized) - 1, -1, -1):
        if normalized[index].get("role") == "assistant":
            return list(normalized[:index]), normalized[index].get("content", "")
    return [], ""


def _normalize_messages(messages: list[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip().lower() or "user"
        content = _content_to_text(message.get("content", "")).strip()
        if content:
            normalized.append({"role": role, "content": content})
    return normalized


def _flatten_prompt_messages(messages: list[dict[str, str]]) -> str:
    lines = [f"{message['role']}: {message['content']}" for message in messages]
    lines.append("assistant: ")
    return "\n".join(lines)


def _flatten_messages(messages: list[Any]) -> str:
    if not isinstance(messages, list):
        return ""
    return _flatten_normalized_messages(_normalize_messages(messages))


def _flatten_normalized_messages(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{message['role']}: {message['content']}" for message in messages
    ).strip()


def _content_to_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or item.get("content") or "")
            if isinstance(item, dict)
            else str(item)
            for item in content
        )
    return str(content or "")


def _load_token_sequences(path: Path | None) -> list[list[int]]:
    return [record.tokens for record in _load_token_sequence_records(path)]


def _load_token_sequence_records(path: Path | None) -> list[TokenSequenceRecord]:
    if path is None or not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    sequences: list[TokenSequenceRecord] = []
    for item in payload.get("requests", []):
        token_ids: Any = None
        min_next_index = 1
        if isinstance(item, dict):
            token_ids = item.get("token_ids") or item.get("tokens")
            min_next_index = _coerce_positive_int(item.get("min_next_index"), 1)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            token_payload = item[1]
            if isinstance(token_payload, dict):
                token_ids = token_payload.get("tokens") or token_payload.get(
                    "token_ids"
                )
                min_next_index = _coerce_positive_int(
                    token_payload.get("min_next_index"), 1
                )
            else:
                token_ids = token_payload
        if not isinstance(token_ids, list):
            continue
        parsed: list[int] = []
        for token_id in token_ids:
            try:
                parsed.append(int(token_id))
            except (TypeError, ValueError):
                continue
        if len(parsed) >= 2:
            sequences.append(
                TokenSequenceRecord(tokens=parsed, min_next_index=min_next_index)
            )
    return sequences


def _filter_token_sequences(
    token_sequences: list[list[int]], *, vocab_size: int
) -> list[list[int]]:
    return [
        record.tokens
        for record in _filter_token_sequence_records(
            [TokenSequenceRecord(tokens=token_ids) for token_ids in token_sequences],
            vocab_size=vocab_size,
        )
    ]


def _filter_token_sequence_records(
    token_sequences: list[TokenSequenceRecord], *, vocab_size: int
) -> list[TokenSequenceRecord]:
    records: list[TokenSequenceRecord] = []
    for record in token_sequences:
        valid = [token_id for token_id in record.tokens if 0 <= token_id < vocab_size]
        if len(valid) >= 2:
            records.append(
                TokenSequenceRecord(
                    tokens=valid,
                    min_next_index=max(1, min(record.min_next_index, len(valid) - 1)),
                )
            )
    return records


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _normalize_token_source_mode(value: str | None) -> str:
    normalized = str(value or "text").strip().lower().replace("-", "_")
    aliases = {
        "token": "tokens",
        "token_sequences": "tokens",
        "traffic_tokens": "tokens",
        "suffix": "tokens",
        "suffix_tokens": "tokens",
        "dataset": "text",
        "traffic_text": "text",
        "completion": "text",
        "completion_text": "text",
        "both": "mixed",
        "all": "mixed",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"auto", "text", "tokens", "mixed"}:
        return "text"
    return normalized


def _training_data_source(
    *,
    staged_finetune: bool,
    use_text_records: bool,
    use_token_sequences: bool,
    primary_records: list[TrainingRecord],
    extra_records: list[TrainingRecord],
    token_sequence_records: list[TokenSequenceRecord],
) -> str:
    data_source_parts: list[str] = []
    if staged_finetune:
        if use_text_records and primary_records:
            data_source_parts.append(_pretrain_text_data_source(primary_records))
        if use_text_records and extra_records:
            data_source_parts.append("finetune_traffic_text")
        if use_token_sequences and token_sequence_records:
            data_source_parts.append("finetune_traffic_token_sequences")
    else:
        if use_text_records and (primary_records or extra_records):
            data_source_parts.append("completion_text")
        if use_token_sequences and token_sequence_records:
            data_source_parts.append("traffic_token_sequences")
    return "+".join(data_source_parts) or "none"


def _source_example_count(
    *,
    staged_finetune: bool,
    use_text_records: bool,
    use_token_sequences: bool,
    primary_records: list[TrainingRecord],
    extra_records: list[TrainingRecord],
    token_sequence_records: list[TokenSequenceRecord],
) -> int:
    if staged_finetune:
        return (
            (len(primary_records) if use_text_records else 0)
            + (len(extra_records) if use_text_records else 0)
            + (len(token_sequence_records) if use_token_sequences else 0)
        )
    return (
        (len(primary_records) + len(extra_records) if use_text_records else 0)
        + (len(token_sequence_records) if use_token_sequences else 0)
    )


def _pretrain_text_data_source(records: list[TrainingRecord]) -> str:
    if not records:
        return "pretrain_completion_text"
    teacher_rows = sum(1 for record in records if record.teacher_distilled)
    if teacher_rows == len(records):
        return "pretrain_teacher_completion_text"
    if teacher_rows > 0:
        return "pretrain_mixed_teacher_completion_text"
    return "pretrain_completion_text"


def _build_training_stages(
    config: TrainerConfig,
    *,
    examples: list[tuple[list[int], list[int]]],
    pretrain_examples: list[tuple[list[int], list[int]]],
    finetune_examples: list[tuple[list[int], list[int]]],
    data_source: str,
) -> tuple[
    list[TrainingStage],
    list[tuple[list[int], list[int]]],
    list[tuple[list[int], list[int]]],
]:
    if not config.staged_finetune:
        train_examples, quality_examples = _split_quality_examples(
            examples,
            max_eval_examples=config.quality_eval_examples,
            min_train_examples=config.min_examples,
        )
        return (
            [
                TrainingStage(
                    name="train",
                    data_source=data_source,
                    examples=train_examples,
                    planned_steps=_planned_max_steps(config, len(train_examples)),
                    promote_checkpoints=True,
                )
            ],
            train_examples,
            quality_examples,
        )

    if finetune_examples:
        pretrain_train_examples = list(pretrain_examples)
        finetune_train_examples, quality_examples = _split_quality_examples(
            finetune_examples,
            max_eval_examples=config.quality_eval_examples,
            min_train_examples=config.min_examples,
        )
    else:
        pretrain_train_examples, quality_examples = _split_quality_examples(
            pretrain_examples,
            max_eval_examples=config.quality_eval_examples,
            min_train_examples=config.min_examples,
        )
        finetune_train_examples = []

    stages: list[TrainingStage] = []
    pretrain_data_source = _stage_data_source(data_source, prefix="pretrain_")
    finetune_planned_steps = (
        _planned_stage_steps(config, "finetune", len(finetune_train_examples))
        if finetune_train_examples
        else 0
    )
    pretrain_planned_steps = (
        _planned_stage_steps(config, "pretrain", len(pretrain_train_examples))
        if pretrain_train_examples
        else 0
    )
    if pretrain_train_examples and finetune_train_examples:
        pretrain_planned_steps = _cap_pretrain_steps_for_live_finetune(
            config, pretrain_planned_steps, finetune_planned_steps
        )
    if pretrain_train_examples:
        stages.append(
            TrainingStage(
                name="pretrain",
                data_source=pretrain_data_source,
                examples=pretrain_train_examples,
                planned_steps=pretrain_planned_steps,
                promote_checkpoints=not bool(finetune_train_examples),
            )
        )
    if finetune_train_examples:
        stages.append(
            TrainingStage(
                name="finetune",
                data_source="finetune_traffic",
                examples=finetune_train_examples,
                planned_steps=finetune_planned_steps,
                promote_checkpoints=True,
            )
        )
    train_examples = pretrain_train_examples + finetune_train_examples
    return stages, train_examples, quality_examples


def _stage_data_source(data_source: str, *, prefix: str) -> str:
    for part in str(data_source or "").split("+"):
        if part.startswith(prefix):
            return part
    if prefix == "pretrain_":
        return "pretrain_completion_text"
    return prefix.rstrip("_")


def _planned_stage_steps(config: TrainerConfig, stage: str, num_examples: int) -> int:
    if num_examples <= 0:
        return 0
    if stage == "pretrain":
        max_steps = config.pretrain_max_steps
        epochs = config.pretrain_epochs
    elif stage == "finetune":
        max_steps = config.finetune_max_steps
        epochs = config.finetune_epochs
    else:
        max_steps = config.max_steps
        epochs = config.train_epochs
    if max_steps > 0:
        return max_steps
    examples_per_epoch = max(0.0, float(num_examples) * max(0.0, epochs))
    return max(1, math.ceil(examples_per_epoch / max(1, config.batch_size)))


def _cap_pretrain_steps_for_live_finetune(
    config: TrainerConfig, pretrain_steps: int, finetune_steps: int
) -> int:
    ratio = max(0.0, float(config.pretrain_live_step_ratio or 0.0))
    if pretrain_steps <= 0 or finetune_steps <= 0 or ratio <= 0.0:
        return pretrain_steps
    cap = max(1, math.ceil(float(finetune_steps) * ratio))
    return min(pretrain_steps, cap)


def _planned_max_steps(config: TrainerConfig, num_examples: int) -> int:
    if config.max_steps > 0:
        return config.max_steps
    examples_per_epoch = max(0.0, float(num_examples) * max(0.0, config.train_epochs))
    return max(1, math.ceil(examples_per_epoch / max(1, config.batch_size)))


def _planned_train_steps(
    config: TrainerConfig, planned_max_steps: int, restored_step: int
) -> int:
    if config.max_steps > 0:
        return planned_max_steps
    if config.auto_full_pass_each_run:
        return planned_max_steps
    if restored_step <= 0:
        return planned_max_steps
    return max(1, planned_max_steps - restored_step)


def _training_recipe_id(data_source: str, config: TrainerConfig) -> str:
    return "|".join(
        [
            data_source,
            f"token_source_mode={_normalize_token_source_mode(config.token_source_mode)}",
            f"staged_finetune={bool(config.staged_finetune)}",
            f"max_seq_len={max(1, int(config.max_seq_len))}",
            f"hidden_size={max(1, int(config.hidden_size))}",
            f"adapter_architecture={config.adapter_architecture}",
            f"spec_tokens={max(1, int(config.num_speculative_tokens))}",
            f"max_examples={int(config.max_examples_per_text)}",
            f"example_stride={max(1, int(config.example_stride))}",
        ]
    )


def _iter_training_batches(
    examples: list[tuple[list[int], list[int]]],
    *,
    batch_size: int,
    max_steps: int,
) -> Any:
    if not examples:
        return
    batch_size = max(1, batch_size)
    shuffled = list(examples)
    random.shuffle(shuffled)
    index = 0
    for _step in range(max(0, max_steps)):
        if index >= len(shuffled):
            random.shuffle(shuffled)
            index = 0
        batch = shuffled[index : index + batch_size]
        index += len(batch)
        if batch:
            yield batch


def _build_examples_from_training_records(
    tokenizer: Any,
    records: list[TrainingRecord],
    max_seq_len: int,
    num_speculative_tokens: int,
    label_pad_token_id: int = 0,
    allow_partial_labels: bool = False,
    max_examples_per_text: int = 4,
    example_stride: int = 1,
) -> list[tuple[list[int], list[int]]]:
    examples: list[tuple[list[int], list[int]]] = []
    prompt_token_cache: dict[tuple[Any, ...], list[int]] = {}
    completion_token_cache: dict[str, list[int]] = {}
    text_token_cache: dict[str, list[int]] = {}
    progress_every = _env_int("VLLM_JETSPEC_BUILD_EXAMPLES_LOG_EVERY", 2500)
    for record_index, record in enumerate(records):
        if record.has_completion_boundary:
            prompt_key = _training_prompt_cache_key(record)
            prompt_ids = prompt_token_cache.get(prompt_key)
            if prompt_ids is None:
                prompt_ids = _tokenize_training_prompt(tokenizer, record)
                prompt_token_cache[prompt_key] = prompt_ids
            completion_ids = completion_token_cache.get(record.completion)
            if completion_ids is None:
                completion_ids = _tokenize_text_to_ids(tokenizer, record.completion)
                completion_token_cache[record.completion] = completion_ids
            token_ids = prompt_ids + completion_ids
            min_label_start = len(prompt_ids)
        else:
            token_ids = text_token_cache.get(record.text)
            if token_ids is None:
                token_ids = _tokenize_text_to_ids(tokenizer, record.text)
                text_token_cache[record.text] = token_ids
            min_label_start = 1
        examples.extend(
            _build_examples_from_single_token_sequence(
                token_ids=token_ids,
                min_label_start=min_label_start,
                max_seq_len=max_seq_len,
                num_speculative_tokens=num_speculative_tokens,
                label_pad_token_id=label_pad_token_id,
                allow_partial_labels=allow_partial_labels,
                max_examples_per_sequence=max_examples_per_text,
                example_stride=example_stride,
                stride_offset=record_index,
            )
        )
        if progress_every > 0 and (record_index + 1) % progress_every == 0:
            logger.info(
                "JetSpec trainer build progress records=%d/%d examples=%d prompt_cache=%d completion_cache=%d",
                record_index + 1,
                len(records),
                len(examples),
                len(prompt_token_cache),
                len(completion_token_cache),
            )
    return examples


def _training_prompt_cache_key(record: TrainingRecord) -> tuple[Any, ...]:
    if record.prompt_messages:
        return (
            "messages",
            tuple(
                (message.get("role", ""), message.get("content", ""))
                for message in record.prompt_messages
            ),
        )
    return ("text", record.prompt)


def _tokenize_training_prompt(tokenizer: Any, record: TrainingRecord) -> list[int]:
    if record.prompt_messages:
        messages = [dict(message) for message in record.prompt_messages]
        if messages:
            try:
                return _coerce_token_ids(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=_env_bool_any(
                            (
                                "VLLM_JETSPEC_FORCE_THINKING",
                                "VLLM_DFLASH_FORCE_THINKING",
                            ),
                            True,
                        ),
                    ),
                    tokenizer=tokenizer,
                )
            except TypeError:
                return _coerce_token_ids(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                    ),
                    tokenizer=tokenizer,
                )
            except Exception:
                logger.debug("Falling back to text prompt tokenization", exc_info=True)
    if not record.prompt:
        return []
    return _tokenize_text_to_ids(tokenizer, record.prompt)


def _tokenize_text_to_ids(tokenizer: Any, text: str) -> list[int]:
    tokenized = tokenizer(text, add_special_tokens=False)
    input_ids = getattr(tokenized, "input_ids", tokenized)
    if isinstance(input_ids, str):
        return []
    return _coerce_token_ids(input_ids)


def _coerce_token_ids(value: Any, *, tokenizer: Any | None = None) -> list[int]:
    if isinstance(value, str):
        if tokenizer is None:
            return []
        return _tokenize_text_to_ids(tokenizer, value)
    input_ids = getattr(value, "input_ids", value)
    if isinstance(input_ids, str):
        if tokenizer is None:
            return []
        return _tokenize_text_to_ids(tokenizer, input_ids)
    if not isinstance(input_ids, (list, tuple)):
        return []
    token_ids: list[int] = []
    for token_id in input_ids:
        try:
            token_ids.append(int(token_id))
        except (TypeError, ValueError):
            continue
    return token_ids


def _build_examples(
    tokenizer: Any,
    texts: list[str],
    max_seq_len: int,
    num_speculative_tokens: int,
    label_pad_token_id: int = 0,
    allow_partial_labels: bool = False,
    max_examples_per_text: int = 4,
    example_stride: int = 1,
) -> list[tuple[list[int], list[int]]]:
    examples: list[tuple[list[int], list[int]]] = []
    for text_index, text in enumerate(texts):
        token_ids = _tokenize_text_to_ids(tokenizer, text)
        examples.extend(
            _build_examples_from_single_token_sequence(
                token_ids=token_ids,
                min_label_start=1,
                max_seq_len=max_seq_len,
                num_speculative_tokens=num_speculative_tokens,
                label_pad_token_id=label_pad_token_id,
                allow_partial_labels=allow_partial_labels,
                max_examples_per_sequence=max_examples_per_text,
                example_stride=example_stride,
                stride_offset=text_index,
            )
        )
    return examples


def _build_examples_from_token_sequence_records(
    *,
    token_sequences: list[TokenSequenceRecord],
    max_seq_len: int,
    num_speculative_tokens: int,
    label_pad_token_id: int = 0,
    allow_partial_labels: bool = False,
    max_examples_per_sequence: int = 4,
    example_stride: int = 1,
) -> list[tuple[list[int], list[int]]]:
    examples: list[tuple[list[int], list[int]]] = []
    for sequence_index, record in enumerate(token_sequences):
        examples.extend(
            _build_examples_from_single_token_sequence(
                token_ids=record.tokens,
                min_label_start=record.min_next_index,
                max_seq_len=max_seq_len,
                num_speculative_tokens=num_speculative_tokens,
                label_pad_token_id=label_pad_token_id,
                allow_partial_labels=allow_partial_labels,
                max_examples_per_sequence=max_examples_per_sequence,
                example_stride=example_stride,
                stride_offset=sequence_index,
            )
        )
    return examples


def _build_examples_from_token_sequences(
    *,
    token_sequences: list[list[int]],
    max_seq_len: int,
    num_speculative_tokens: int,
    label_pad_token_id: int = 0,
    allow_partial_labels: bool = False,
    max_examples_per_sequence: int = 4,
    example_stride: int = 1,
) -> list[tuple[list[int], list[int]]]:
    return _build_examples_from_token_sequence_records(
        token_sequences=[
            TokenSequenceRecord(tokens=list(token_ids), min_next_index=1)
            for token_ids in token_sequences
        ],
        max_seq_len=max_seq_len,
        num_speculative_tokens=num_speculative_tokens,
        label_pad_token_id=label_pad_token_id,
        allow_partial_labels=allow_partial_labels,
        max_examples_per_sequence=max_examples_per_sequence,
        example_stride=example_stride,
    )


def _build_examples_from_single_token_sequence(
    *,
    token_ids: list[int],
    min_label_start: int,
    max_seq_len: int,
    num_speculative_tokens: int,
    label_pad_token_id: int = 0,
    allow_partial_labels: bool = False,
    max_examples_per_sequence: int = 4,
    example_stride: int = 1,
    stride_offset: int = 0,
) -> list[tuple[list[int], list[int]]]:
    token_ids = _coerce_token_ids(token_ids)
    if len(token_ids) <= 1:
        return []
    if allow_partial_labels:
        max_start = len(token_ids) - 1
    else:
        if len(token_ids) <= num_speculative_tokens + 1:
            return []
        max_start = len(token_ids) - num_speculative_tokens
    min_start = max(1, int(min_label_start))
    examples: list[tuple[list[int], list[int]]] = []
    for start in _training_starts(
        max_start,
        max_examples_per_sequence,
        example_stride=example_stride,
        stride_offset=stride_offset,
        min_start=min_start,
    ):
        context = token_ids[max(0, start - max_seq_len) : start]
        labels = token_ids[start : min(len(token_ids), start + num_speculative_tokens)]
        if context and labels:
            if len(labels) < num_speculative_tokens:
                labels = labels + [label_pad_token_id] * (
                    num_speculative_tokens - len(labels)
                )
            examples.append((context, labels[:num_speculative_tokens]))
    return examples


def _training_starts(
    max_start: int,
    max_examples_per_text: int,
    *,
    example_stride: int = 1,
    stride_offset: int = 0,
    min_start: int = 1,
) -> list[int]:
    min_start = max(1, int(min_start))
    if max_start < min_start:
        return []
    stride = max(1, int(example_stride))
    if max_examples_per_text <= 0:
        first = 1 + (max(0, int(stride_offset)) % stride)
        if first < min_start:
            first += math.ceil((min_start - first) / stride) * stride
        starts = list(range(first, max_start + 1, stride))
        if not starts or starts[-1] != max_start:
            starts.append(max_start)
        return starts
    if max_start - min_start + 1 <= max_examples_per_text:
        return list(range(min_start, max_start + 1))
    if max_examples_per_text == 1:
        return [max_start]
    if stride > 1:
        first = 1 + (max(0, int(stride_offset)) % stride)
        if first < min_start:
            first += math.ceil((min_start - first) / stride) * stride
        candidates = list(range(first, max_start + 1, stride))
        if not candidates or candidates[-1] != max_start:
            candidates.append(max_start)
        if len(candidates) <= max_examples_per_text:
            return candidates
        return _evenly_sample_starts(candidates, max_examples_per_text)
    return _evenly_sample_starts(
        list(range(min_start, max_start + 1)), max_examples_per_text
    )


def _evenly_sample_starts(candidates: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= len(candidates):
        return list(candidates)
    if count == 1:
        return [candidates[-1]]
    last_index = len(candidates) - 1
    return sorted(
        {
            candidates[round(index * last_index / (count - 1))]
            for index in range(count)
        }
    )


def _collate(
    batch: list[tuple[list[int], list[int]]], *, pad_token_id: int, device: Any
):
    import torch

    max_len = max(len(context) for context, _labels in batch)
    max_label_len = max(len(labels) for _context, labels in batch)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    for context, labels in batch:
        pad = [pad_token_id] * (max_len - len(context))
        input_rows.append(pad + context)
        label_rows.append(
            labels + [pad_token_id] * (max(0, max_label_len - len(labels)))
        )
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(label_rows, dtype=torch.long, device=device),
    )


def _model_forward(model: Any, input_ids: Any, labels: Any | None = None) -> Any:
    if labels is None:
        return model(input_ids)
    try:
        return model(input_ids, labels=labels)
    except TypeError:
        return model(input_ids)


def _evaluate_draft_quality(
    model: Any,
    examples: list[tuple[list[int], list[int]]],
    *,
    pad_token_id: int,
    device: Any,
    batch_size: int,
    max_examples: int,
) -> dict[str, Any]:
    import torch

    selected = _select_quality_examples(examples, max_examples=max_examples)
    if not selected:
        return {
            "examples": 0,
            "top1_accuracy": 0.0,
            "exact_prefix_mean": 0.0,
        }

    was_training = bool(getattr(model, "training", False))
    model.eval()
    correct_tokens = 0
    total_tokens = 0
    prefix_total = 0
    with torch.inference_mode():
        for start in range(0, len(selected), max(1, batch_size)):
            batch = selected[start : start + max(1, batch_size)]
            input_ids, labels = _collate(batch, pad_token_id=pad_token_id, device=device)
            predictions = _model_forward(model, input_ids, labels).argmax(dim=-1)
            valid = labels.ne(pad_token_id)
            matches = predictions.eq(labels) & valid
            correct_tokens += int(matches.sum().item())
            total_tokens += int(valid.sum().item())
            for row_matches, row_valid in zip(matches.tolist(), valid.tolist()):
                prefix = 0
                for matched, valid_token in zip(row_matches, row_valid):
                    if not valid_token:
                        break
                    if not matched:
                        break
                    prefix += 1
                prefix_total += prefix
    if was_training:
        model.train()
    return {
        "examples": len(selected),
        "top1_accuracy": correct_tokens / total_tokens if total_tokens else 0.0,
        "exact_prefix_mean": prefix_total / len(selected),
    }


def _evaluate_latest_checkpoint_quality(
    checkpoint_dir: Path,
    examples: list[tuple[list[int], list[int]]],
    *,
    pad_token_id: int,
    device: Any,
    batch_size: int,
    max_examples: int,
    expected_data_source: str | None = None,
    expected_training_recipe: str | None = None,
    reset_on_mismatch: bool = True,
) -> dict[str, Any] | None:
    checkpoint = _latest_checkpoint(checkpoint_dir)
    if checkpoint is None or not checkpoint.exists():
        return None
    try:
        payload = load_checkpoint(checkpoint, map_location="cpu")
        if reset_on_mismatch and (expected_data_source or expected_training_recipe):
            metadata = payload.get("metadata") or {}
            checkpoint_data_source = metadata.get("data_source")
            if expected_data_source and checkpoint_data_source != expected_data_source:
                logger.info(
                    "Skipping latest JetSpec checkpoint quality eval for %s because data_source=%s expected=%s",
                    checkpoint,
                    checkpoint_data_source or "missing",
                    expected_data_source,
                )
                return None
            checkpoint_training_recipe = metadata.get("training_recipe")
            if (
                expected_training_recipe
                and not _training_recipe_matches(
                    checkpoint_training_recipe, expected_training_recipe
                )
            ):
                logger.info(
                    "Skipping latest JetSpec checkpoint quality eval for %s because training_recipe=%s expected=%s",
                    checkpoint,
                    checkpoint_training_recipe or "missing",
                    expected_training_recipe,
                )
                return None
        config_payload = payload.get("config")
        if not config_payload:
            return None
        expected_spec_tokens = len(examples[0][1]) if examples else 0
        checkpoint_spec_tokens = int(config_payload.get("num_speculative_tokens") or 0)
        if expected_spec_tokens and checkpoint_spec_tokens != expected_spec_tokens:
            logger.info(
                "Skipping latest JetSpec checkpoint quality eval for %s because spec_tokens=%d expected=%d",
                checkpoint,
                checkpoint_spec_tokens,
                expected_spec_tokens,
            )
            return None
        model = JetSpecAdapter(JetSpecAdapterConfig(**config_payload))
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)
        quality = _evaluate_draft_quality(
            model,
            examples,
            pad_token_id=pad_token_id,
            device=device,
            batch_size=batch_size,
            max_examples=max_examples,
        )
        quality["checkpoint_path"] = str(checkpoint)
        quality["checkpoint_step"] = int(payload.get("step") or 0)
        model.to("cpu")
        return quality
    except Exception:
        logger.exception("Failed to evaluate latest JetSpec checkpoint %s", checkpoint)
        return None


def _select_quality_examples(
    examples: list[tuple[list[int], list[int]]], *, max_examples: int
) -> list[tuple[list[int], list[int]]]:
    if max_examples <= 0 or len(examples) <= max_examples:
        return list(examples)
    if max_examples == 1:
        return [examples[-1]]
    last_index = len(examples) - 1
    indices = sorted(
        {
            round(index * last_index / (max_examples - 1))
            for index in range(max_examples)
        }
    )
    return [examples[index] for index in indices]


def _split_quality_examples(
    examples: list[tuple[list[int], list[int]]],
    *,
    max_eval_examples: int,
    min_train_examples: int,
) -> tuple[list[tuple[list[int], list[int]]], list[tuple[list[int], list[int]]]]:
    if max_eval_examples <= 0 or len(examples) <= min_train_examples + 1:
        return list(examples), []
    max_eval = min(max_eval_examples, max(1, len(examples) // 5))
    eval_count = min(max_eval, len(examples) - min_train_examples)
    if eval_count <= 0:
        return list(examples), []
    if eval_count == 1:
        eval_indices = {len(examples) - 1}
    else:
        last_index = len(examples) - 1
        eval_indices = {
            round(index * last_index / (eval_count - 1))
            for index in range(eval_count)
        }
    train_examples = [
        example for index, example in enumerate(examples) if index not in eval_indices
    ]
    quality_examples = [
        example for index, example in enumerate(examples) if index in eval_indices
    ]
    if len(train_examples) < min_train_examples:
        return list(examples), []
    return train_examples, quality_examples


def _should_promote_quality(
    baseline_quality: dict[str, Any] | None,
    final_quality: dict[str, Any] | None,
    *,
    min_delta: float,
) -> bool:
    if not final_quality:
        return False
    if int(final_quality.get("examples") or 0) <= 0:
        return False
    final_score = float(final_quality.get("exact_prefix_mean") or 0.0)
    if not baseline_quality:
        return final_score + 1e-9 >= max(0.0, min_delta)
    baseline_score = float(baseline_quality.get("exact_prefix_mean") or 0.0)
    return final_score + 1e-9 >= baseline_score + min_delta


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    latest_checkpoint = _checkpoint_from_latest_json(checkpoint_dir / "latest.json")
    if latest_checkpoint is not None and latest_checkpoint.exists():
        return latest_checkpoint
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-step-*.pt"))
    return checkpoints[-1] if checkpoints else None


def _max_checkpoint_step(checkpoint_dir: Path) -> int:
    max_step = 0
    try:
        checkpoints = list(checkpoint_dir.glob("checkpoint-step-*.pt"))
    except OSError:
        return 0
    for checkpoint in checkpoints:
        stem = checkpoint.stem
        try:
            step = int(stem.rsplit("-", 1)[-1])
        except (TypeError, ValueError):
            continue
        max_step = max(max_step, step)
    return max_step


def _checkpoint_from_latest_json(latest_path: Path) -> Path | None:
    if not latest_path.exists():
        return None
    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    checkpoint = payload.get("checkpoint")
    if not checkpoint:
        return None
    path = Path(str(checkpoint))
    if not path.is_absolute():
        path = latest_path.parent / path
    return path


def _restore_latest(
    checkpoint_dir: Path,
    model: Any,
    optimizer: Any | None,
    *,
    expected_data_source: str | None = None,
    expected_training_recipe: str | None = None,
    reset_on_mismatch: bool = True,
) -> int:
    checkpoint = _latest_checkpoint(checkpoint_dir)
    if checkpoint is None:
        return 0
    try:
        payload = load_checkpoint(checkpoint, map_location="cpu")
        if expected_data_source and reset_on_mismatch:
            metadata = payload.get("metadata") or {}
            checkpoint_data_source = metadata.get("data_source")
            if checkpoint_data_source != expected_data_source:
                logger.info(
                    "JetSpec trainer ignoring checkpoint %s because data_source=%s expected=%s",
                    checkpoint,
                    checkpoint_data_source or "missing",
                    expected_data_source,
                )
                return 0
            checkpoint_training_recipe = metadata.get("training_recipe")
            if (
                expected_training_recipe
                and not _training_recipe_matches(
                    checkpoint_training_recipe, expected_training_recipe
                )
            ):
                logger.info(
                    "JetSpec trainer ignoring checkpoint %s because training_recipe=%s expected=%s",
                    checkpoint,
                    checkpoint_training_recipe or "missing",
                    expected_training_recipe,
                )
                return 0
        model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
    except Exception:
        return 0
    return int(payload.get("step") or 0)


def _training_recipe_matches(
    checkpoint_training_recipe: Any, expected_training_recipe: str
) -> bool:
    checkpoint_recipe = str(checkpoint_training_recipe or "")
    expected_recipe = str(expected_training_recipe or "")
    if not expected_recipe:
        return True
    if checkpoint_recipe == expected_recipe:
        return True
    if not checkpoint_recipe:
        return False
    checkpoint_source, checkpoint_values = _parse_training_recipe(checkpoint_recipe)
    expected_source, expected_values = _parse_training_recipe(expected_recipe)
    if checkpoint_source != expected_source:
        return False
    for key, expected_value in expected_values.items():
        if checkpoint_values.get(key) != expected_value:
            return False
    return True


def _parse_training_recipe(recipe: str) -> tuple[str, dict[str, str]]:
    source = ""
    values: dict[str, str] = {}
    for index, part in enumerate(str(recipe or "").split("|")):
        if not part:
            continue
        if "=" not in part:
            if index == 0:
                source = part
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return source, values


def _save(
    config: TrainerConfig,
    model: Any,
    optimizer: Any,
    step: int,
    metadata: dict[str, Any],
    *,
    promote_latest: bool = True,
) -> None:
    checkpoint_path = config.checkpoint_dir / f"checkpoint-step-{step:08d}.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer=optimizer if config.save_optimizer_state else None,
        step=step,
        metadata={"model_name": config.model_name, **metadata},
    )
    if promote_latest:
        latest_path = config.checkpoint_dir / "latest.json"
        latest_payload = {
            "checkpoint": checkpoint_path.name,
            "checkpoint_path": str(checkpoint_path),
            "step": step,
            "saved_at": time.time(),
            "model_name": config.model_name,
            "num_speculative_tokens": config.num_speculative_tokens,
            "quality": metadata.get("quality"),
        }
        tmp_latest_path = latest_path.with_suffix(latest_path.suffix + ".tmp")
        tmp_latest_path.write_text(
            json.dumps(latest_payload, indent=2) + "\n", encoding="utf-8"
        )
        tmp_latest_path.replace(latest_path)
    _prune_checkpoints(config)


def _prune_checkpoints(config: TrainerConfig) -> None:
    _cleanup_stale_tmp_checkpoints(config.checkpoint_dir)
    if config.keep_last_checkpoints <= 0:
        return
    checkpoints = sorted(
        config.checkpoint_dir.glob("checkpoint-step-*.pt"),
        key=lambda checkpoint: checkpoint.stat().st_mtime,
    )
    latest_checkpoint = _checkpoint_from_latest_json(config.checkpoint_dir / "latest.json")
    keep_paths = set(checkpoints[-config.keep_last_checkpoints :])
    if latest_checkpoint is not None:
        keep_paths.add(latest_checkpoint)
    for checkpoint in checkpoints:
        if checkpoint in keep_paths:
            continue
        try:
            checkpoint.unlink()
        except FileNotFoundError:
            pass


def _cleanup_stale_tmp_checkpoints(
    checkpoint_dir: Path, *, max_age_seconds: float = 3600.0
) -> None:
    cutoff = time.time() - max_age_seconds
    for tmp_checkpoint in checkpoint_dir.glob("checkpoint-step-*.pt.tmp"):
        try:
            if tmp_checkpoint.stat().st_mtime < cutoff:
                tmp_checkpoint.unlink()
        except FileNotFoundError:
            pass


def _stop_requested(stop_file: Path | None) -> bool:
    return _TERMINATION_REQUESTED or bool(stop_file and stop_file.exists())


def _write_metric(config: TrainerConfig, row: dict[str, Any]) -> None:
    if not _is_primary_rank():
        return
    if config.metrics_path is None:
        return
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": time.time(), **row}
    if config.run_id:
        payload["run_id"] = config.run_id
    with config.metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_bool_any(names: tuple[str, ...], default: bool) -> bool:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw != "":
            return raw.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw != "":
            return raw
    return default


def _split_env_paths(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.replace(":", ",").split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interruptible JIT JetSpec adapter trainer"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument(
        "--extra-data-path",
        action="append",
        default=[],
        help="Additional JSONL training data path. Can be repeated.",
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--suffix-cache-path",
        default=_env_first(
            "VLLM_JETSPEC_SUFFIX_CACHE_PATH", "VLLM_DFLASH_SUFFIX_CACHE_PATH"
        ),
    )
    parser.add_argument(
        "--token-source-mode",
        default=_env_first(
            "VLLM_JETSPEC_TRAIN_TOKEN_SOURCE_MODE",
            "VLLM_DFLASH_TRAIN_TOKEN_SOURCE_MODE",
            default="text",
        ),
        choices=("auto", "text", "tokens", "mixed"),
        help=(
            "Use JSONL prompt/completion text, persisted traffic token sequences, "
            "or both for JetSpec adapter training."
        ),
    )
    parser.add_argument(
        "--staged-finetune",
        action=argparse.BooleanOptionalAction,
        default=_env_bool_any(
            ("VLLM_JETSPEC_STAGED_FINETUNE", "VLLM_DFLASH_STAGED_FINETUNE"),
            False,
        ),
        help=(
            "Train in two phases: pretrain on the primary dataset, then finetune "
            "on extra live traffic and token traces. Final promotion is based on "
            "the finetune/live holdout when live data exists."
        ),
    )
    parser.add_argument(
        "--device",
        default=_env_first(
            "VLLM_JETSPEC_TRAIN_DEVICE", "VLLM_DFLASH_TRAIN_DEVICE", default="cuda:0"
        ),
    )
    parser.add_argument(
        "--parallel-mode",
        default=_env_first(
            "VLLM_JETSPEC_TRAIN_PARALLEL_MODE",
            "VLLM_DFLASH_TRAIN_PARALLEL_MODE",
            default="auto",
        ),
        choices=("auto", "none", "ddp", "data_parallel", "distributed"),
        help=(
            "Training parallelism for the lightweight JetSpec adapter. In auto "
            "mode, torchrun-launched workers use DDP; single-process runs stay "
            "on one device."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(
            _env_first(
                "VLLM_JETSPEC_TRAIN_STEPS", "VLLM_DFLASH_TRAIN_STEPS", default="0"
            )
        ),
    )
    parser.add_argument(
        "--pretrain-max-steps",
        type=int,
        default=int(
            _env_first(
                "VLLM_JETSPEC_PRETRAIN_MAX_STEPS",
                "VLLM_DFLASH_PRETRAIN_MAX_STEPS",
                default="0",
            )
        ),
    )
    parser.add_argument(
        "--finetune-max-steps",
        type=int,
        default=int(
            _env_first(
                "VLLM_JETSPEC_FINETUNE_MAX_STEPS",
                "VLLM_DFLASH_FINETUNE_MAX_STEPS",
                default="0",
            )
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(
            _env_first("VLLM_JETSPEC_BATCH_SIZE", "VLLM_DFLASH_BATCH_SIZE", default="4")
        ),
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=int(
            _env_first(
                "VLLM_JETSPEC_MAX_SEQ_LEN", "VLLM_DFLASH_MAX_SEQ_LEN", default="256"
            )
        ),
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=int(
            _env_first(
                "VLLM_JETSPEC_HIDDEN_SIZE", "VLLM_DFLASH_HIDDEN_SIZE", default="32"
            )
        ),
    )
    parser.add_argument(
        "--adapter-architecture",
        default=_env_first(
            "VLLM_JETSPEC_ADAPTER_ARCHITECTURE",
            "VLLM_DFLASH_ADAPTER_ARCHITECTURE",
            default="pooled_gru",
        ),
        choices=("pooled_gru", "autoregressive_gru"),
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=int(
            _env_first(
                "VLLM_JETSPEC_NUM_SPECULATIVE_TOKENS",
                "VLLM_DFLASH_NUM_SPECULATIVE_TOKENS",
                default="8",
            )
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(_env_first("VLLM_JETSPEC_LR", "VLLM_DFLASH_LR", default="0.01")),
    )
    parser.add_argument(
        "--train-epochs",
        type=float,
        default=float(
            _env_first(
                "VLLM_JETSPEC_TRAIN_EPOCHS", "VLLM_DFLASH_TRAIN_EPOCHS", default="1"
            )
        ),
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=float,
        default=float(
            _env_first(
                "VLLM_JETSPEC_PRETRAIN_EPOCHS",
                "VLLM_DFLASH_PRETRAIN_EPOCHS",
                "VLLM_JETSPEC_TRAIN_EPOCHS",
                "VLLM_DFLASH_TRAIN_EPOCHS",
                default="1",
            )
        ),
    )
    parser.add_argument(
        "--finetune-epochs",
        type=float,
        default=float(
            _env_first(
                "VLLM_JETSPEC_FINETUNE_EPOCHS",
                "VLLM_DFLASH_FINETUNE_EPOCHS",
                "VLLM_JETSPEC_TRAIN_EPOCHS",
                "VLLM_DFLASH_TRAIN_EPOCHS",
                default="1",
            )
        ),
    )
    parser.add_argument(
        "--pretrain-live-step-ratio",
        type=float,
        default=float(
            _env_first(
                "VLLM_JETSPEC_PRETRAIN_LIVE_STEP_RATIO",
                "VLLM_DFLASH_PRETRAIN_LIVE_STEP_RATIO",
                default="0",
            )
        ),
        help=(
            "When staged finetune has live traffic examples, cap pretrain steps "
            "to this ratio of the planned finetune steps."
        ),
    )
    parser.add_argument(
        "--max-examples-per-traffic",
        type=int,
        default=int(
            _env_first(
                "VLLM_JETSPEC_MAX_EXAMPLES_PER_TRAFFIC",
                "VLLM_DFLASH_MAX_EXAMPLES_PER_TRAFFIC",
                default="4",
            )
        ),
    )
    parser.add_argument(
        "--example-stride",
        type=int,
        default=int(
            _env_first(
                "VLLM_JETSPEC_EXAMPLE_STRIDE",
                "VLLM_DFLASH_EXAMPLE_STRIDE",
                default="1",
            )
        ),
        help=(
            "When max examples per traffic row is 0, train every Nth live-like "
            "next-token position instead of only sparse sampled offsets."
        ),
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=int(
            _env_first(
                "VLLM_JETSPEC_MIN_EXAMPLES", "VLLM_DFLASH_MIN_EXAMPLES", default="8"
            )
        ),
    )
    parser.add_argument(
        "--auto-full-pass-each-run",
        action=argparse.BooleanOptionalAction,
        default=_env_bool_any(
            (
                "VLLM_JETSPEC_AUTO_FULL_PASS_EACH_RUN",
                "VLLM_DFLASH_AUTO_FULL_PASS_EACH_RUN",
            ),
            True,
        ),
    )
    parser.add_argument(
        "--reset-on-data-source-change",
        action=argparse.BooleanOptionalAction,
        default=_env_bool_any(
            (
                "VLLM_JETSPEC_RESET_ON_DATA_SOURCE_CHANGE",
                "VLLM_DFLASH_RESET_ON_DATA_SOURCE_CHANGE",
            ),
            False,
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=int(
            _env_first(
                "VLLM_JETSPEC_CHECKPOINT_EVERY",
                "VLLM_DFLASH_CHECKPOINT_EVERY",
                default="16",
            )
        ),
    )
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=int(
            _env_first(
                "VLLM_JETSPEC_KEEP_LAST_CHECKPOINTS",
                "VLLM_DFLASH_KEEP_LAST_CHECKPOINTS",
                default="2",
            )
        ),
    )
    parser.add_argument(
        "--save-optimizer-state",
        action="store_true",
        default=_env_bool_any(
            ("VLLM_JETSPEC_SAVE_OPTIMIZER_STATE", "VLLM_DFLASH_SAVE_OPTIMIZER_STATE"),
            False,
        ),
    )
    parser.add_argument(
        "--quality-gate",
        action=argparse.BooleanOptionalAction,
        default=_env_bool_any(
            ("VLLM_JETSPEC_QUALITY_GATE", "VLLM_DFLASH_QUALITY_GATE"),
            True,
        ),
    )
    parser.add_argument(
        "--quality-eval-examples",
        type=int,
        default=int(
            _env_first(
                "VLLM_JETSPEC_QUALITY_EVAL_EXAMPLES",
                "VLLM_DFLASH_QUALITY_EVAL_EXAMPLES",
                default="128",
            )
        ),
    )
    parser.add_argument(
        "--quality-min-delta",
        type=float,
        default=float(
            _env_first(
                "VLLM_JETSPEC_QUALITY_MIN_DELTA",
                "VLLM_DFLASH_QUALITY_MIN_DELTA",
                default="0",
            )
        ),
    )
    parser.add_argument(
        "--promote-intermediate-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=_env_bool_any(
            (
                "VLLM_JETSPEC_PROMOTE_INTERMEDIATE_CHECKPOINTS",
                "VLLM_DFLASH_PROMOTE_INTERMEDIATE_CHECKPOINTS",
            ),
            False,
        ),
        help=(
            "When quality gating is enabled, evaluate checkpoint-every saves and "
            "promote improved JetSpec checkpoints during long online training runs."
        ),
    )
    parser.add_argument(
        "--stop-file",
        default=_env_first("VLLM_JETSPEC_STOP_FILE", "VLLM_DFLASH_STOP_FILE"),
    )
    parser.add_argument(
        "--metrics-path",
        default=_env_first("VLLM_JETSPEC_METRICS_PATH", "VLLM_DFLASH_METRICS_PATH"),
    )
    parser.add_argument(
        "--run-id",
        default=_env_first(
            "VLLM_JETSPEC_TRAIN_RUN_ID", "VLLM_DFLASH_TRAIN_RUN_ID", default=""
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=int(
            _env_first(
                "VLLM_JETSPEC_TRAIN_LOG_EVERY",
                "VLLM_DFLASH_TRAIN_LOG_EVERY",
                default="1",
            )
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_training(
        TrainerConfig(
            model_name=args.model_name,
            data_path=Path(args.data_path),
            checkpoint_dir=Path(args.checkpoint_dir),
            extra_data_paths=tuple(
                Path(path)
                for path in [
                    *args.extra_data_path,
                    *_split_env_paths(
                        _env_first(
                            "VLLM_JETSPEC_EXTRA_TRAIN_DATA_PATHS",
                            "VLLM_DFLASH_EXTRA_TRAIN_DATA_PATHS",
                            default="",
                        )
                    ),
                ]
                if path
            ),
            suffix_cache_path=Path(args.suffix_cache_path)
            if args.suffix_cache_path
            else None,
            token_source_mode=args.token_source_mode,
            staged_finetune=args.staged_finetune,
            device=args.device,
            parallel_mode=args.parallel_mode,
            max_steps=args.max_steps,
            pretrain_max_steps=args.pretrain_max_steps,
            finetune_max_steps=args.finetune_max_steps,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            hidden_size=args.hidden_size,
            adapter_architecture=args.adapter_architecture,
            num_speculative_tokens=args.num_speculative_tokens,
            learning_rate=args.learning_rate,
            train_epochs=args.train_epochs,
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
            pretrain_live_step_ratio=args.pretrain_live_step_ratio,
            max_examples_per_text=args.max_examples_per_traffic,
            example_stride=args.example_stride,
            min_examples=args.min_examples,
            auto_full_pass_each_run=args.auto_full_pass_each_run,
            reset_on_data_source_change=args.reset_on_data_source_change,
            checkpoint_every=args.checkpoint_every,
            keep_last_checkpoints=args.keep_last_checkpoints,
            save_optimizer_state=args.save_optimizer_state,
            quality_gate_enabled=args.quality_gate,
            quality_eval_examples=args.quality_eval_examples,
            quality_min_delta=args.quality_min_delta,
            promote_intermediate_checkpoints=args.promote_intermediate_checkpoints,
            stop_file=Path(args.stop_file) if args.stop_file else None,
            metrics_path=Path(args.metrics_path) if args.metrics_path else None,
            run_id=args.run_id,
            log_every=args.log_every,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
