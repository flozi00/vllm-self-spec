"""Colocated vLLM inference and LoRA training supervisor.

One container, two GPU halves:

- vLLM serves the base model directly on the public inference port using one
  half of the visible GPUs, started with ``--enable-lora`` and runtime LoRA
  updating so adapters can be swapped without a restart.
- A FastAPI control plane on a second port accepts SFT/DPO/KTO training jobs
  whose samples are provided in the request body. Jobs run sequentially in a
  trainer subprocess pinned to the other half of the GPUs.
- After each training job the exported PEFT adapter is loaded into the running
  vLLM server via ``/v1/load_lora_adapter`` (fast weight sync). Clients reach
  the tuned weights by requesting the adapter model name.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query

logger = logging.getLogger("vllm_colocate.app")

TRAINER_SCRIPT = Path(__file__).with_name("lora_trainer.py")
SUPERVISOR_ENV_PREFIX = "VLLM_COLOCATE_"
TRAIN_KINDS = ("sft", "dpo", "kto")
# vLLM's LoRAConfig only accepts these --max-lora-rank values.
VLLM_ALLOWED_MAX_LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)
JOB_OPTION_FIELDS: dict[str, type] = {
    "max_steps": int,
    "train_epochs": float,
    "batch_size": int,
    "gradient_accumulation_steps": int,
    "max_seq_len": int,
    "learning_rate": float,
    "weight_decay": float,
    "max_grad_norm": float,
    "dpo_beta": float,
    "kto_beta": float,
    "kto_desirable_weight": float,
    "kto_undesirable_weight": float,
    "loss_vocab_sample_size": int,
}


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_optional_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    if raw.strip() == "":
        return None
    return int(raw)


def _round_up_max_lora_rank(rank: int) -> int:
    for allowed in VLLM_ALLOWED_MAX_LORA_RANKS:
        if allowed >= rank:
            return allowed
    return VLLM_ALLOWED_MAX_LORA_RANKS[-1]


def _default_tensor_parallel_size(gpu_count: int) -> int:
    """Largest power of two <= gpu_count: attention-head counts are almost
    always divisible by powers of two, while e.g. TP=3 fails outright."""
    size = 1
    while size * 2 <= max(1, gpu_count):
        size *= 2
    return size


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "--", value).strip("-._")
    if cleaned:
        return cleaned[:96]
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _extra_arg_present(extra_args: str, option: str) -> bool:
    try:
        parts = shlex.split(extra_args or "")
    except ValueError:
        return False
    return option in parts or any(part.startswith(f"{option}=") for part in parts)


def _detect_gpu_ids() -> tuple[str, ...]:
    raw = os.getenv("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        normalized = raw.strip().lower()
        if normalized in {"", "void", "none"}:
            return ()
        if normalized != "all":
            return _split_csv(raw)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0:
            ids = _split_csv(result.stdout.replace("\n", ","))
            if ids:
                return ids
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import torch

        count = torch.cuda.device_count()
    except Exception:
        return ()
    return tuple(str(index) for index in range(count))


def _partition_gpus() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Split visible GPUs between inference and training.

    Returns ``(all_gpus, inference_gpus, training_gpus)``. Explicit
    ``VLLM_COLOCATE_INFERENCE_GPUS`` / ``VLLM_COLOCATE_TRAINING_GPUS`` lists
    win; otherwise inference gets the first half (rounded up) and training the
    rest. A single-GPU host shares the one device between both roles.
    """
    detected = _detect_gpu_ids()
    inference = _split_csv(os.getenv("VLLM_COLOCATE_INFERENCE_GPUS"))
    training = _split_csv(os.getenv("VLLM_COLOCATE_TRAINING_GPUS"))
    if detected:
        for name, override in (
            ("VLLM_COLOCATE_INFERENCE_GPUS", inference),
            ("VLLM_COLOCATE_TRAINING_GPUS", training),
        ):
            unknown = [gpu for gpu in override if gpu not in detected]
            if unknown:
                raise RuntimeError(
                    f"{name} entries {unknown!r} are not among the detected GPU "
                    f"ids {list(detected)!r}; use the same identifiers (indices "
                    "from nvidia-smi, or the tokens in this process's "
                    "CUDA_VISIBLE_DEVICES)."
                )
    if inference and training:
        all_gpus = detected or tuple(dict.fromkeys(inference + training))
        return all_gpus, inference, training
    if inference:
        all_gpus = detected or inference
        remaining = tuple(g for g in all_gpus if g not in inference)
        return all_gpus, inference, remaining or inference
    if training:
        all_gpus = detected or training
        remaining = tuple(g for g in all_gpus if g not in training)
        return all_gpus, remaining or training, training
    if not detected:
        return (), (), ()
    if len(detected) == 1:
        return detected, detected, detected
    split = math.ceil(len(detected) / 2)
    return detected, detected[:split], detected[split:]


@dataclass(frozen=True)
class Settings:
    model_name: str
    served_model_name: str
    adapter_name: str
    inference_host: str
    inference_port: int
    api_host: str
    api_port: int
    data_dir: Path
    checkpoint_dir: Path
    jobs_dir: Path
    metrics_path: Path
    all_gpus: tuple[str, ...]
    inference_gpus: tuple[str, ...]
    training_gpus: tuple[str, ...]
    gpu_memory_utilization: float
    tensor_parallel_size: int
    max_model_len: int | None
    max_num_seqs: int | None
    max_loras: int
    max_lora_rank: int
    trust_remote_code: bool
    vllm_extra_args: str
    ready_timeout_seconds: float
    cancel_grace_seconds: float
    sync_on_startup: bool

    @property
    def inference_base_url(self) -> str:
        host = self.inference_host if self.inference_host != "0.0.0.0" else "127.0.0.1"
        return f"http://{host}:{self.inference_port}"

    @property
    def gpus_shared(self) -> bool:
        return bool(self.inference_gpus) and self.inference_gpus == self.training_gpus

    @classmethod
    def from_env(cls) -> "Settings":
        model_name = _env_str("VLLM_COLOCATE_MODEL", "Qwen/Qwen3-8B")
        served_model_name = _env_str(
            "VLLM_COLOCATE_SERVED_MODEL_NAME", model_name
        )
        # Resolve to absolute paths so latest.json entries (written by the
        # trainer relative to the checkpoint dir) never double-join.
        data_dir = Path(
            _env_str("VLLM_COLOCATE_DATA_DIR", "/data/vllm_colocate")
        ).resolve()
        checkpoint_dir = Path(
            _env_str(
                "VLLM_COLOCATE_CHECKPOINT_DIR",
                str(data_dir / "checkpoints" / _safe_slug(model_name)),
            )
        ).resolve()
        all_gpus, inference_gpus, training_gpus = _partition_gpus()
        gpus_shared = bool(inference_gpus) and inference_gpus == training_gpus
        lora_r = _env_int("VLLM_COLOCATE_LORA_R", 16)
        return cls(
            model_name=model_name,
            served_model_name=served_model_name,
            adapter_name=_env_str(
                "VLLM_COLOCATE_LORA_ADAPTER_NAME", f"{served_model_name}-lora"
            ),
            inference_host=_env_str("VLLM_COLOCATE_INFERENCE_HOST", "0.0.0.0"),
            inference_port=_env_int("VLLM_COLOCATE_INFERENCE_PORT", 8000),
            api_host=_env_str("VLLM_COLOCATE_API_HOST", "0.0.0.0"),
            api_port=_env_int("VLLM_COLOCATE_API_PORT", 8001),
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            jobs_dir=Path(
                _env_str("VLLM_COLOCATE_JOBS_DIR", str(data_dir / "jobs"))
            ).resolve(),
            metrics_path=Path(
                _env_str(
                    "VLLM_COLOCATE_METRICS_PATH",
                    str(data_dir / "trainer_metrics.jsonl"),
                )
            ).resolve(),
            all_gpus=all_gpus,
            inference_gpus=inference_gpus,
            training_gpus=training_gpus,
            gpu_memory_utilization=_env_float(
                "VLLM_COLOCATE_GPU_MEMORY_UTILIZATION",
                0.45 if gpus_shared else 0.90,
            ),
            tensor_parallel_size=_env_int(
                "VLLM_COLOCATE_TENSOR_PARALLEL_SIZE",
                _default_tensor_parallel_size(len(inference_gpus)),
            ),
            max_model_len=_env_optional_int("VLLM_COLOCATE_MAX_MODEL_LEN", None),
            max_num_seqs=_env_optional_int("VLLM_COLOCATE_MAX_NUM_SEQS", None),
            max_loras=_env_int("VLLM_COLOCATE_MAX_LORAS", 1),
            max_lora_rank=_env_int(
                "VLLM_COLOCATE_MAX_LORA_RANK",
                _round_up_max_lora_rank(max(16, lora_r)),
            ),
            trust_remote_code=_env_bool("VLLM_COLOCATE_TRUST_REMOTE_CODE", True),
            vllm_extra_args=_env_str("VLLM_COLOCATE_VLLM_EXTRA_ARGS", ""),
            ready_timeout_seconds=_env_float(
                "VLLM_COLOCATE_READY_TIMEOUT_SECONDS", 1800.0
            ),
            cancel_grace_seconds=_env_float(
                "VLLM_COLOCATE_CANCEL_GRACE_SECONDS", 30.0
            ),
            sync_on_startup=_env_bool("VLLM_COLOCATE_SYNC_ON_STARTUP", True),
        )


class VllmProcess:
    """Launches and talks to the vLLM OpenAI server on the inference GPUs."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.process: subprocess.Popen[Any] | None = None
        self.ready = False

    def status(self) -> dict[str, Any]:
        process = self.process
        return {
            "running": process is not None and process.poll() is None,
            "ready": self.ready,
            "pid": process.pid if process is not None else None,
            "returncode": process.returncode if process is not None else None,
            "base_url": self.settings.inference_base_url,
        }

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.ready = False
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._command()
        env = self._child_env()
        logger.info(
            "Starting vLLM on GPUs [%s]: %s",
            ",".join(self.settings.inference_gpus) or "-",
            shlex.join(cmd),
        )
        self.process = subprocess.Popen(cmd, env=env)

    def terminate(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()

    def _child_env(self) -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(SUPERVISOR_ENV_PREFIX)
        }
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        env.setdefault("PYTHONUNBUFFERED", "1")
        if self.settings.inference_gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(self.settings.inference_gpus)
        return env

    def _command(self) -> list[str]:
        settings = self.settings
        cmd = [
            "vllm",
            "serve",
            settings.model_name,
            "--served-model-name",
            settings.served_model_name,
            "--host",
            settings.inference_host,
            "--port",
            str(settings.inference_port),
            "--gpu-memory-utilization",
            str(settings.gpu_memory_utilization),
        ]
        extra_args = settings.vllm_extra_args
        if not _extra_arg_present(extra_args, "--enable-lora"):
            cmd.append("--enable-lora")
        if not _extra_arg_present(extra_args, "--max-loras"):
            cmd.extend(["--max-loras", str(settings.max_loras)])
        if not _extra_arg_present(extra_args, "--max-lora-rank"):
            cmd.extend(["--max-lora-rank", str(settings.max_lora_rank)])
        if settings.tensor_parallel_size > 1 and not _extra_arg_present(
            extra_args, "--tensor-parallel-size"
        ):
            cmd.extend(
                ["--tensor-parallel-size", str(settings.tensor_parallel_size)]
            )
        if settings.trust_remote_code:
            cmd.append("--trust-remote-code")
        if settings.max_model_len:
            cmd.extend(["--max-model-len", str(settings.max_model_len)])
        if settings.max_num_seqs:
            cmd.extend(["--max-num-seqs", str(settings.max_num_seqs)])
        if extra_args:
            cmd.extend(shlex.split(extra_args))
        return cmd

    async def wait_ready(self, timeout_seconds: float | None = None) -> None:
        if timeout_seconds is None:
            timeout_seconds = self.settings.ready_timeout_seconds
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.monotonic() < deadline:
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError(
                        f"vLLM exited with code {self.process.returncode}"
                    )
                try:
                    response = await client.get(
                        f"{self.settings.inference_base_url}/v1/models"
                    )
                    if response.status_code < 500:
                        self.ready = True
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(2)
        raise TimeoutError("Timed out waiting for vLLM to become ready")

    async def is_ready(self) -> bool:
        if self.process is None or self.process.poll() is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.settings.inference_base_url}/v1/models"
                )
                return response.status_code < 500
        except httpx.HTTPError:
            return False

    async def load_lora_adapter(
        self, *, adapter_name: str, adapter_path: Path
    ) -> tuple[bool, str | None]:
        payload = {"lora_name": adapter_name, "lora_path": str(adapter_path)}
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                await client.post(
                    f"{self.settings.inference_base_url}/v1/unload_lora_adapter",
                    json={"lora_name": adapter_name},
                )
            except httpx.HTTPError:
                pass
            try:
                response = await client.post(
                    f"{self.settings.inference_base_url}/v1/load_lora_adapter",
                    json=payload,
                )
            except httpx.HTTPError as exc:
                return False, f"load_lora_adapter request failed: {exc}"
        if response.status_code < 400:
            return True, None
        return (
            False,
            f"vLLM rejected adapter load status={response.status_code} "
            f"body={response.text[:500]}",
        )

    async def loaded_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.settings.inference_base_url}/v1/models"
                )
                if response.status_code >= 400:
                    return []
                payload = response.json()
        except (httpx.HTTPError, ValueError):
            return []
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return []
        return [
            str(item.get("id"))
            for item in data
            if isinstance(item, dict) and item.get("id")
        ]


@dataclass
class TrainingJob:
    id: str
    kind: str
    sample_count: int
    data_path: Path
    stop_file: Path
    options: dict[str, Any]
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    returncode: int | None = None
    error: str | None = None
    cancel_requested: bool = False
    step_before: int | None = None
    step_after: int | None = None
    sync: dict[str, Any] | None = None
    done_event: asyncio.Event = field(default_factory=asyncio.Event)

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "sample_count": self.sample_count,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
            "step_before": self.step_before,
            "step_after": self.step_after,
            "sync": self.sync,
            "options": self.options,
            "data_path": str(self.data_path),
        }


def _validate_sample(kind: str, sample: Any) -> str | None:
    if not isinstance(sample, dict):
        return "sample must be a JSON object"
    messages = sample.get("messages")
    if messages is not None and (
        not isinstance(messages, list)
        or not all(isinstance(message, dict) for message in messages)
    ):
        return "messages must be a list of objects"
    has_prompt = bool(sample.get("prompt")) or bool(sample.get("messages"))
    if kind == "sft":
        if sample.get("messages") or sample.get("text"):
            return None
        if sample.get("prompt") is not None and sample.get("completion") is not None:
            return None
        return "sft sample needs messages, text, or prompt+completion"
    if kind == "dpo":
        if not has_prompt:
            return "dpo sample needs prompt or messages"
        if not sample.get("chosen") or not sample.get("rejected"):
            return "dpo sample needs chosen and rejected"
        return None
    if kind == "kto":
        if not has_prompt:
            return "kto sample needs prompt or messages"
        if not sample.get("completion"):
            return "kto sample needs completion"
        label = sample.get("label", sample.get("desirable"))
        if not isinstance(label, bool):
            return "kto sample needs boolean label (or desirable)"
        return None
    return f"unknown task kind {kind!r}"


def _extract_samples(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("samples", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if payload:
            return [payload]
    return []


def _extract_options(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    raw = payload.get("options")
    if not isinstance(raw, dict):
        return {}
    options: dict[str, Any] = {}
    errors: list[str] = []
    for key, value in raw.items():
        caster = JOB_OPTION_FIELDS.get(key)
        if caster is None:
            errors.append(f"unknown option {key!r}")
            continue
        try:
            options[key] = caster(value)
        except (TypeError, ValueError):
            errors.append(f"option {key!r} must be {caster.__name__}")
            continue
        if (
            key in ("batch_size", "gradient_accumulation_steps", "max_seq_len")
            and options[key] < 1
        ):
            errors.append(f"option {key!r} must be >= 1")
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors[:5]})
    return options


def _write_job_rows(data_path: Path, kind: str, samples: list[Any]) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    with data_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            row = {
                **sample,
                "task": kind,
                "id": sample.get("id") or str(uuid.uuid4()),
                "ts": now,
            }
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


_TAIL_LINE_BUDGET_BYTES = 8192


def _tail_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    limit = max(1, limit)
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            budget = min(size, limit * _TAIL_LINE_BUDGET_BYTES)
            handle.seek(size - budget)
            chunk = handle.read(budget)
    except OSError:
        return []
    lines = chunk.decode("utf-8", errors="replace").splitlines()
    if budget < size and lines:
        lines = lines[1:]  # drop the first, likely partial, line
    rows: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


class TrainingManager:
    """Serializes training jobs onto the training GPUs and syncs adapters."""

    def __init__(self, settings: Settings, vllm: VllmProcess):
        self.settings = settings
        self.vllm = vllm
        self.jobs: dict[str, TrainingJob] = {}
        self.job_order: list[str] = []
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.worker_task: asyncio.Task[None] | None = None
        self.current_process: asyncio.subprocess.Process | None = None
        self.current_job_id: str | None = None
        self.sync_lock = asyncio.Lock()
        self.active_adapter_path = settings.checkpoint_dir / "active_adapter.json"
        self._cancel_tasks: set[asyncio.Task[None]] = set()

    # ---------------------------------------------------------------- queue

    def start(self) -> None:
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        # Capture the in-flight process/job BEFORE cancelling the worker:
        # _run_job's finally block clears both while the cancellation unwinds.
        process = self.current_process
        job = self.jobs.get(self.current_job_id) if self.current_job_id else None
        if self.worker_task is not None:
            self.worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.worker_task
            self.worker_task = None
        if process is not None and process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=10)
            if process.returncode is None:
                process.kill()
        if job is not None and not job.done_event.is_set():
            job.status = "canceled"
            job.error = "supervisor shutdown"
            job.finished_at = time.time()
            job.done_event.set()

    async def submit(
        self, kind: str, samples: list[Any], options: dict[str, Any]
    ) -> TrainingJob:
        errors = []
        for index, sample in enumerate(samples):
            error = _validate_sample(kind, sample)
            if error:
                errors.append(f"sample[{index}]: {error}")
            if len(errors) >= 5:
                break
        if errors:
            raise HTTPException(status_code=400, detail={"errors": errors})
        job_id = f"{kind}-{uuid.uuid4().hex[:12]}"
        data_path = self.settings.jobs_dir / f"{job_id}.jsonl"
        await asyncio.to_thread(_write_job_rows, data_path, kind, samples)
        job = TrainingJob(
            id=job_id,
            kind=kind,
            sample_count=len(samples),
            data_path=data_path,
            stop_file=self.settings.jobs_dir / f"{job_id}.stop",
            options=options,
        )
        self.jobs[job.id] = job
        self.job_order.append(job.id)
        self.queue.put_nowait(job.id)
        logger.info(
            "Queued %s training job %s with %d samples", kind, job.id, len(samples)
        )
        return job

    async def cancel(self, job_id: str) -> TrainingJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job id")
        if job.status in {"succeeded", "failed", "canceled"}:
            return job
        if job.returncode is not None:
            # The trainer already exited; the job is only finishing its
            # adapter sync. Too late to cancel.
            return job
        job.cancel_requested = True
        if job.status == "queued":
            job.status = "canceled"
            job.finished_at = time.time()
            job.done_event.set()
            return job
        try:
            job.stop_file.parent.mkdir(parents=True, exist_ok=True)
            job.stop_file.touch()
        except OSError:
            logger.warning("Could not write stop file for job %s", job_id)
        task = asyncio.create_task(self._escalate_cancel(job))
        self._cancel_tasks.add(task)
        task.add_done_callback(self._cancel_tasks.discard)
        return job

    async def _escalate_cancel(self, job: TrainingJob) -> None:
        deadline = time.monotonic() + max(0.0, self.settings.cancel_grace_seconds)
        while time.monotonic() < deadline:
            if job.done_event.is_set():
                return
            await asyncio.sleep(1.0)
        process = self.current_process
        if self.current_job_id == job.id and process is not None:
            if process.returncode is None:
                logger.info("Terminating training job %s after cancel grace", job.id)
                process.terminate()
                await asyncio.sleep(5.0)
                if process.returncode is None:
                    process.kill()

    # --------------------------------------------------------------- worker

    async def _worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.status != "queued":
                continue
            try:
                await self._run_job(job)
            except Exception as exc:
                logger.exception("Training job %s crashed in supervisor", job.id)
                job.status = "failed"
                job.error = str(exc)
                job.finished_at = time.time()
                job.done_event.set()

    async def _run_job(self, job: TrainingJob) -> None:
        job.status = "running"
        job.started_at = time.time()
        self.current_job_id = job.id
        job.step_before = self.latest_checkpoint_step()
        cmd = self._trainer_command(job)
        env = self._trainer_env()
        logger.info(
            "Starting training job %s on GPUs [%s]: %s",
            job.id,
            ",".join(self.settings.training_gpus) or "-",
            shlex.join(cmd),
        )
        try:
            process = await asyncio.create_subprocess_exec(*cmd, env=env)
        except OSError as exc:
            job.status = "failed"
            job.error = f"could not start trainer: {exc}"
            job.finished_at = time.time()
            self.current_job_id = None
            job.done_event.set()
            return
        self.current_process = process
        try:
            job.returncode = await process.wait()
        finally:
            self.current_process = None
            self.current_job_id = None
            with contextlib.suppress(OSError):
                job.stop_file.unlink(missing_ok=True)
        job.step_after = self.latest_checkpoint_step()
        checkpoint_advanced = (
            job.step_after is not None
            and job.step_after > (job.step_before or 0)
        )
        if checkpoint_advanced:
            job.sync = await self.sync_adapter()
        if job.cancel_requested:
            job.status = "canceled"
        elif job.returncode == 0:
            job.status = "succeeded"
        else:
            job.status = "failed"
            if job.returncode == 2:
                job.error = (
                    "trainer reported a missing training dependency "
                    "(see trainer logs)"
                )
            elif job.returncode == 3:
                job.error = (
                    "trainer built no trainable examples from the submitted "
                    "samples (see trainer logs)"
                )
            else:
                job.error = f"trainer exited with code {job.returncode}"
        job.finished_at = time.time()
        logger.info(
            "Training job %s finished status=%s returncode=%s steps=%s->%s synced=%s",
            job.id,
            job.status,
            job.returncode,
            job.step_before,
            job.step_after,
            bool(job.sync and job.sync.get("ok")),
        )
        job.done_event.set()

    def _trainer_command(self, job: TrainingJob) -> list[str]:
        settings = self.settings
        cmd = [
            sys.executable,
            str(TRAINER_SCRIPT),
            "--model-name",
            settings.model_name,
            "--data-path",
            str(job.data_path),
            "--checkpoint-dir",
            str(settings.checkpoint_dir),
            "--adapter-name",
            settings.adapter_name,
            "--task-mix",
            job.kind,
            "--run-id",
            job.id,
            "--metrics-path",
            str(settings.metrics_path),
            "--stop-file",
            str(job.stop_file),
        ]
        for key, value in job.options.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        return cmd

    def _trainer_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("OMP_NUM_THREADS", "1")
        if self.settings.training_gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(self.settings.training_gpus)
        return env

    # ---------------------------------------------------------- adapter sync

    def latest_checkpoint(self) -> dict[str, Any]:
        return _read_json_file(self.settings.checkpoint_dir / "latest.json")

    def latest_checkpoint_step(self) -> int | None:
        latest = self.latest_checkpoint()
        try:
            step = latest.get("step")
            return int(step) if step is not None else None
        except (TypeError, ValueError):
            return None

    def resolve_adapter_path(self) -> Path | None:
        latest = self.latest_checkpoint()
        raw = latest.get("adapter_path")
        if not raw:
            return None
        path = Path(str(raw))
        if not path.is_absolute():
            path = self.settings.checkpoint_dir / path
        if not path.is_dir():
            return None
        return path

    async def sync_adapter(self) -> dict[str, Any]:
        async with self.sync_lock:
            adapter_name = self.settings.adapter_name
            # Bounded wait: if vLLM is still loading, the supervisor loop
            # re-syncs the adapter as soon as it becomes ready.
            try:
                await self.vllm.wait_ready(timeout_seconds=300.0)
            except (RuntimeError, TimeoutError) as exc:
                return {"ok": False, "error": f"vLLM not ready: {exc}"}
            # Resolve AFTER the wait so a checkpoint pruned or replaced in the
            # meantime cannot leave us loading a stale path.
            latest = self.latest_checkpoint()
            adapter_path = self.resolve_adapter_path()
            if adapter_path is None:
                return {
                    "ok": False,
                    "error": "no exported adapter found in latest.json",
                }
            ok, error = await self.vllm.load_lora_adapter(
                adapter_name=adapter_name, adapter_path=adapter_path
            )
            result = {
                "ok": ok,
                "adapter_name": adapter_name,
                "adapter_path": str(adapter_path),
                "step": latest.get("step"),
                "synced_at": time.time(),
            }
            if not ok:
                result["error"] = error
                logger.warning("Adapter sync failed: %s", error)
                # The stable adapter name was already unloaded; roll back to
                # the last known-good adapter so inference keeps serving
                # tuned weights.
                previous = self.active_adapter()
                previous_path = previous.get("adapter_path")
                if (
                    previous_path
                    and previous_path != str(adapter_path)
                    and Path(previous_path).is_dir()
                ):
                    rolled_back, rollback_error = await self.vllm.load_lora_adapter(
                        adapter_name=adapter_name,
                        adapter_path=Path(previous_path),
                    )
                    result["rolled_back"] = rolled_back
                    if rollback_error:
                        result["rollback_error"] = rollback_error
                    logger.warning(
                        "Rolled back adapter %s to %s: %s",
                        adapter_name,
                        previous_path,
                        rolled_back,
                    )
                return result
            try:
                self.active_adapter_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.active_adapter_path.with_suffix(".json.tmp")
                tmp.write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8"
                )
                tmp.replace(self.active_adapter_path)
            except OSError:
                logger.warning("Could not persist active_adapter.json", exc_info=True)
            logger.info(
                "Synced adapter %s (step %s) into vLLM from %s",
                adapter_name,
                latest.get("step"),
                adapter_path,
            )
            return result

    def active_adapter(self) -> dict[str, Any]:
        return _read_json_file(self.active_adapter_path)

    # --------------------------------------------------------------- status

    def jobs_payload(self, *, limit: int = 50) -> list[dict[str, Any]]:
        ids = self.job_order[-max(1, limit):]
        return [self.jobs[job_id].payload() for job_id in reversed(ids)]

    def status(self) -> dict[str, Any]:
        latest = self.latest_checkpoint()
        return {
            "queued_jobs": sum(
                1 for job in self.jobs.values() if job.status == "queued"
            ),
            "running_job": self.current_job_id,
            "total_jobs": len(self.jobs),
            "latest_checkpoint": {
                "step": latest.get("step"),
                "adapter_path": latest.get("adapter_path"),
                "saved_at": latest.get("saved_at"),
            },
            "active_adapter": self.active_adapter(),
        }


settings = Settings.from_env()
vllm = VllmProcess(settings)
manager = TrainingManager(settings, vllm)


async def _supervise_vllm() -> None:
    """Keep vLLM alive; restore the last adapter after each (re)start."""
    backoff = 5.0
    while True:
        process = vllm.process
        if process is None or process.poll() is not None:
            if process is not None:
                logger.warning(
                    "vLLM exited with code %s; restarting in %.0fs",
                    process.returncode,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(300.0, backoff * 2)
            try:
                vllm.start()
            except Exception:
                logger.exception("Could not start vLLM")
                await asyncio.sleep(backoff)
                backoff = min(300.0, backoff * 2)
                continue
        if not vllm.ready:
            try:
                await vllm.wait_ready()
            except RuntimeError:
                # Process exited while waiting; the restart branch above
                # applies the backoff on the next iteration.
                logger.exception("vLLM exited before becoming ready")
                continue
            except TimeoutError:
                logger.warning("vLLM still not ready; continuing to wait")
                continue
            backoff = 5.0
            if settings.sync_on_startup and manager.resolve_adapter_path() is not None:
                await manager.sync_adapter()
        await asyncio.sleep(10.0)


@contextlib.asynccontextmanager
async def _lifespan(_: FastAPI):
    manager.start()
    supervisor = asyncio.create_task(_supervise_vllm())
    try:
        yield
    finally:
        supervisor.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await supervisor
        await manager.stop()
        vllm.terminate()


app = FastAPI(title="vllm-colocate", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "vllm": vllm.status(),
    }


@app.get("/status")
async def status() -> dict[str, Any]:
    vllm_status = vllm.status()
    vllm_status["ready"] = await vllm.is_ready()
    vllm_status["models"] = await vllm.loaded_models()
    return {
        "model": settings.model_name,
        "served_model_name": settings.served_model_name,
        "adapter_name": settings.adapter_name,
        "gpus": {
            "all": list(settings.all_gpus),
            "inference": list(settings.inference_gpus),
            "training": list(settings.training_gpus),
            "shared": settings.gpus_shared,
        },
        "vllm": vllm_status,
        "training": manager.status(),
    }


async def _submit_training(
    kind: str, payload: Any, wait: bool, wait_timeout_seconds: float | None
) -> dict[str, Any]:
    samples = _extract_samples(payload)
    if not samples:
        raise HTTPException(status_code=400, detail="no training samples provided")
    options = _extract_options(payload)
    job = await manager.submit(kind, samples, options)
    if wait:
        # asyncio.wait_for already treats None as unbounded and 0 as an
        # immediate done-check.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                job.done_event.wait(), timeout=wait_timeout_seconds
            )
    return job.payload()


@app.post("/train/sft")
async def train_sft(
    payload: Any = Body(...),
    wait: bool = Query(default=False),
    wait_timeout_seconds: float | None = Query(default=None, ge=0),
) -> dict[str, Any]:
    return await _submit_training("sft", payload, wait, wait_timeout_seconds)


@app.post("/train/dpo")
async def train_dpo(
    payload: Any = Body(...),
    wait: bool = Query(default=False),
    wait_timeout_seconds: float | None = Query(default=None, ge=0),
) -> dict[str, Any]:
    return await _submit_training("dpo", payload, wait, wait_timeout_seconds)


@app.post("/train/kto")
async def train_kto(
    payload: Any = Body(...),
    wait: bool = Query(default=False),
    wait_timeout_seconds: float | None = Query(default=None, ge=0),
) -> dict[str, Any]:
    return await _submit_training("kto", payload, wait, wait_timeout_seconds)


@app.get("/train/jobs")
async def train_jobs(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    return {"jobs": manager.jobs_payload(limit=limit)}


@app.get("/train/jobs/{job_id}")
async def train_job(job_id: str) -> dict[str, Any]:
    job = manager.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    return job.payload()


@app.post("/train/jobs/{job_id}/cancel")
async def train_job_cancel(job_id: str) -> dict[str, Any]:
    job = await manager.cancel(job_id)
    return job.payload()


@app.get("/train/metrics")
async def train_metrics(limit: int = Query(default=50, ge=1, le=1000)) -> dict[str, Any]:
    rows = await asyncio.to_thread(_tail_jsonl, settings.metrics_path, limit=limit)
    return {
        "metrics_path": str(settings.metrics_path),
        "rows": rows,
    }


@app.get("/adapter")
async def adapter() -> dict[str, Any]:
    loaded_models = await vllm.loaded_models()
    return {
        "adapter_name": settings.adapter_name,
        "loaded": settings.adapter_name in loaded_models,
        "active": manager.active_adapter(),
        "latest_checkpoint": manager.latest_checkpoint(),
    }


@app.post("/adapter/sync")
async def adapter_sync() -> dict[str, Any]:
    result = await manager.sync_adapter()
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=result)
    return result


def main() -> None:
    logging.basicConfig(
        level=os.getenv("VLLM_COLOCATE_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info(
        "GPU partition: all=[%s] inference=[%s] training=[%s]%s",
        ",".join(settings.all_gpus) or "-",
        ",".join(settings.inference_gpus) or "-",
        ",".join(settings.training_gpus) or "-",
        " (single GPU shared between inference and training)"
        if settings.gpus_shared
        else "",
    )
    if not settings.all_gpus:
        logger.warning(
            "No GPUs detected; starting anyway (vLLM will decide device placement)"
        )
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=os.getenv("VLLM_COLOCATE_UVICORN_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
