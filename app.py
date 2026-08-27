"""vLLM-native SFT: one server, one port, one training route.

This launcher runs vLLM's stock OpenAI-compatible server in-process and
attaches a single extra route to the same app on the same port:

- ``POST /train/sft`` accepts SFT samples in the request body and queues a
  LoRA training job. ``GET /train/sft`` reports training state.
- All visible GPUs serve inference at all times; the trainer subprocess sees
  the same GPUs and coexists in the memory headroom left by
  ``--gpu-memory-utilization`` (default lowered to leave room for training).
- Training steps only run while inference is idle: an in-flight request
  counter on the vLLM app toggles a pause file that the trainer polls
  between steps, so inference always has the GPUs when requests arrive.
- The base model (quantized checkpoints included) stays loaded in vLLM the
  whole time. After each job the exported PEFT adapter is hot-swapped via
  ``/v1/load_lora_adapter`` — vLLM applies it on top of the resident base
  weights, so there is never a base-weight reload.
"""

from __future__ import annotations

import os

# Old vLLM versions register the runtime-LoRA routes at import time of the
# api_server module, and newer ones may cache env lookups — this must be set
# before anything imports vllm.
os.environ.setdefault("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "1")

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Body, HTTPException, Query

logger = logging.getLogger("vllm_colocate.app")

TRAINER_SCRIPT = Path(__file__).with_name("lora_trainer.py")
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


@dataclass(frozen=True)
class Settings:
    model_name: str
    served_model_name: str
    adapter_name: str
    host: str
    port: int
    data_dir: Path
    checkpoint_dir: Path
    jobs_dir: Path
    metrics_path: Path
    pause_file: Path
    gpus: tuple[str, ...]
    gpu_memory_utilization: float
    tensor_parallel_size: int
    max_model_len: int | None
    max_num_seqs: int | None
    max_loras: int
    max_lora_rank: int
    trust_remote_code: bool
    vllm_extra_args: str
    ready_timeout_seconds: float
    idle_grace_seconds: float
    sync_on_startup: bool

    @property
    def base_url(self) -> str:
        host = self.host if self.host not in ("0.0.0.0", "::", "") else "127.0.0.1"
        return f"http://{host}:{self.port}"

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
        lora_r = _env_int("VLLM_COLOCATE_LORA_R", 16)
        gpus = _detect_gpu_ids()
        return cls(
            model_name=model_name,
            served_model_name=served_model_name,
            adapter_name=_env_str(
                "VLLM_COLOCATE_LORA_ADAPTER_NAME", f"{served_model_name}-lora"
            ),
            host=_env_str("VLLM_COLOCATE_HOST", "0.0.0.0"),
            port=_env_int("VLLM_COLOCATE_PORT", 8000),
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
            pause_file=Path(
                _env_str(
                    "VLLM_COLOCATE_PAUSE_FILE", str(data_dir / "trainer.pause")
                )
            ).resolve(),
            gpus=gpus,
            # Inference and training share every GPU, so vLLM must leave
            # memory headroom for the trainer subprocess.
            gpu_memory_utilization=_env_float(
                "VLLM_COLOCATE_GPU_MEMORY_UTILIZATION", 0.45
            ),
            tensor_parallel_size=_env_int(
                "VLLM_COLOCATE_TENSOR_PARALLEL_SIZE",
                _default_tensor_parallel_size(len(gpus)),
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
            idle_grace_seconds=_env_float(
                "VLLM_COLOCATE_IDLE_GRACE_SECONDS", 5.0
            ),
            sync_on_startup=_env_bool("VLLM_COLOCATE_SYNC_ON_STARTUP", True),
        )


# --------------------------------------------------------------------- gate

_GATE_EXCLUDED_PATHS = frozenset(
    {"/v1/load_lora_adapter", "/v1/unload_lora_adapter"}
)
_GATE_IDLE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class InFlightGate:
    """Pure-ASGI middleware counting in-flight inference requests.

    Installed through vLLM's own ``--middleware`` mechanism, so it wraps the
    stock app without patching it. State is class-level: the training gate
    reads it to decide when the trainer may run.
    """

    in_flight = 0
    last_activity = 0.0  # monotonic; 0.0 = no inference request seen yet

    def __init__(self, app: Any):
        self.app = app

    @staticmethod
    def _counts(scope: dict[str, Any]) -> bool:
        if scope.get("method", "GET").upper() in _GATE_IDLE_METHODS:
            return False
        path = scope.get("path", "")
        if path in _GATE_EXCLUDED_PATHS or path.startswith("/train"):
            return False
        return True

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not self._counts(scope):
            await self.app(scope, receive, send)
            return
        cls = InFlightGate
        cls.in_flight += 1
        cls.last_activity = time.monotonic()
        try:
            await self.app(scope, receive, send)
        finally:
            cls.in_flight -= 1
            cls.last_activity = time.monotonic()

    @classmethod
    def busy(cls, idle_grace_seconds: float) -> bool:
        if cls.in_flight > 0:
            return True
        if cls.last_activity == 0.0:
            return False
        return (time.monotonic() - cls.last_activity) < idle_grace_seconds


# ---------------------------------------------------------------- training


@dataclass
class TrainingJob:
    id: str
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
    step_before: int | None = None
    step_after: int | None = None
    sync: dict[str, Any] | None = None
    done_event: asyncio.Event = field(default_factory=asyncio.Event)

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "sample_count": self.sample_count,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "error": self.error,
            "step_before": self.step_before,
            "step_after": self.step_after,
            "sync": self.sync,
            "options": self.options,
            "data_path": str(self.data_path),
        }


def _validate_sample(sample: Any) -> str | None:
    if not isinstance(sample, dict):
        return "sample must be a JSON object"
    messages = sample.get("messages")
    if messages is not None and (
        not isinstance(messages, list)
        or not all(isinstance(message, dict) for message in messages)
    ):
        return "messages must be a list of objects"
    if sample.get("messages") or sample.get("text"):
        return None
    if sample.get("prompt") is not None and sample.get("completion") is not None:
        return None
    return "sft sample needs messages, text, or prompt+completion"


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


def _write_job_rows(data_path: Path, samples: list[Any]) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    with data_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            row = {
                **sample,
                "task": "sft",
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
    """Queues SFT jobs, runs the trainer while inference is idle, and
    hot-swaps the exported adapter into the running vLLM server."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.jobs: dict[str, TrainingJob] = {}
        self.job_order: list[str] = []
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.worker_task: asyncio.Task[None] | None = None
        self.gate_task: asyncio.Task[None] | None = None
        self.current_process: asyncio.subprocess.Process | None = None
        self.current_job_id: str | None = None
        self.sync_lock = asyncio.Lock()
        self.active_adapter_path = settings.checkpoint_dir / "active_adapter.json"
        self.trainer_paused: bool | None = None

    # ---------------------------------------------------------------- queue

    def start(self) -> None:
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())
        if self.gate_task is None or self.gate_task.done():
            self.gate_task = asyncio.create_task(self._gate_loop())

    async def stop(self) -> None:
        # Capture the in-flight process/job BEFORE cancelling the worker:
        # _run_job's finally block clears both while the cancellation unwinds.
        process = self.current_process
        job = self.jobs.get(self.current_job_id) if self.current_job_id else None
        for task in (self.worker_task, self.gate_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self.worker_task = None
        self.gate_task = None
        if process is not None and process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=10)
            if process.returncode is None:
                process.kill()
        if job is not None and not job.done_event.is_set():
            job.status = "canceled"
            job.error = "server shutdown"
            job.finished_at = time.time()
            job.done_event.set()
        with contextlib.suppress(OSError):
            self.settings.pause_file.unlink(missing_ok=True)

    async def submit(
        self, samples: list[Any], options: dict[str, Any]
    ) -> TrainingJob:
        errors = []
        for index, sample in enumerate(samples):
            error = _validate_sample(sample)
            if error:
                errors.append(f"sample[{index}]: {error}")
            if len(errors) >= 5:
                break
        if errors:
            raise HTTPException(status_code=400, detail={"errors": errors})
        job_id = f"sft-{uuid.uuid4().hex[:12]}"
        data_path = self.settings.jobs_dir / f"{job_id}.jsonl"
        await asyncio.to_thread(_write_job_rows, data_path, samples)
        job = TrainingJob(
            id=job_id,
            sample_count=len(samples),
            data_path=data_path,
            stop_file=self.settings.jobs_dir / f"{job_id}.stop",
            options=options,
        )
        self.jobs[job.id] = job
        self.job_order.append(job.id)
        self.queue.put_nowait(job.id)
        logger.info("Queued SFT job %s with %d samples", job.id, len(samples))
        return job

    # ----------------------------------------------------------------- gate

    async def _gate_loop(self) -> None:
        """Mirror inference activity into the trainer pause file.

        The trainer polls the file between steps: present = pause, absent =
        train. Toggled only on transitions so the loop is one stat-free
        comparison most of the time.
        """
        pause_file = self.settings.pause_file
        while True:
            busy = InFlightGate.busy(self.settings.idle_grace_seconds)
            if busy != self.trainer_paused:
                try:
                    if busy:
                        pause_file.parent.mkdir(parents=True, exist_ok=True)
                        pause_file.touch()
                    else:
                        pause_file.unlink(missing_ok=True)
                    self.trainer_paused = busy
                except OSError:
                    logger.warning("Could not update trainer pause file")
            await asyncio.sleep(0.25)

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
        logger.info("Starting training job %s: %s", job.id, shlex.join(cmd))
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
        if job.returncode == 0:
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
            "--run-id",
            job.id,
            "--metrics-path",
            str(settings.metrics_path),
            "--stop-file",
            str(job.stop_file),
            "--pause-file",
            str(settings.pause_file),
        ]
        for key, value in job.options.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        return cmd

    def _trainer_env(self) -> dict[str, str]:
        # The trainer shares every GPU with vLLM: no CUDA_VISIBLE_DEVICES
        # override, just allocator settings that play nice with a neighbor.
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("OMP_NUM_THREADS", "1")
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

    async def wait_server_ready(self, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.monotonic() < deadline:
                try:
                    response = await client.get(f"{self.settings.base_url}/health")
                    if response.status_code == 200:
                        return True
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(2)
        return False

    async def _load_lora_adapter(
        self, *, adapter_name: str, adapter_path: Path
    ) -> tuple[bool, str | None]:
        payload = {"lora_name": adapter_name, "lora_path": str(adapter_path)}
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Unload-then-load works on every vLLM version; a missing name
            # just 404s, which is fine.
            try:
                await client.post(
                    f"{self.settings.base_url}/v1/unload_lora_adapter",
                    json={"lora_name": adapter_name},
                )
            except httpx.HTTPError:
                pass
            try:
                response = await client.post(
                    f"{self.settings.base_url}/v1/load_lora_adapter",
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

    async def sync_adapter(self) -> dict[str, Any]:
        async with self.sync_lock:
            adapter_name = self.settings.adapter_name
            if not await self.wait_server_ready(300.0):
                return {"ok": False, "error": "vLLM server not ready"}
            # Resolve AFTER the wait so a checkpoint pruned or replaced in the
            # meantime cannot leave us loading a stale path.
            latest = self.latest_checkpoint()
            adapter_path = self.resolve_adapter_path()
            if adapter_path is None:
                return {
                    "ok": False,
                    "error": "no exported adapter found in latest.json",
                }
            ok, error = await self._load_lora_adapter(
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
                    rolled_back, rollback_error = await self._load_lora_adapter(
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
            "paused": bool(self.trainer_paused),
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


# ------------------------------------------------------------------- routes

settings = Settings.from_env()
manager = TrainingManager(settings)
train_router = APIRouter()


@train_router.post("/train/sft")
async def train_sft(
    payload: Any = Body(...),
    wait: bool = Query(default=False),
    wait_timeout_seconds: float | None = Query(default=None, ge=0),
) -> dict[str, Any]:
    samples = _extract_samples(payload)
    if not samples:
        raise HTTPException(status_code=400, detail="no training samples provided")
    options = _extract_options(payload)
    job = await manager.submit(samples, options)
    if wait:
        # asyncio.wait_for already treats None as unbounded and 0 as an
        # immediate done-check.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                job.done_event.wait(), timeout=wait_timeout_seconds
            )
    return job.payload()


@train_router.get("/train/sft")
async def train_sft_status(
    limit: int = Query(default=20, ge=1, le=500),
) -> dict[str, Any]:
    return {
        "model": settings.model_name,
        "served_model_name": settings.served_model_name,
        "adapter_name": settings.adapter_name,
        "gpus": list(settings.gpus),
        "inference": {
            "in_flight": InFlightGate.in_flight,
            "busy": InFlightGate.busy(settings.idle_grace_seconds),
            "idle_grace_seconds": settings.idle_grace_seconds,
        },
        "training": manager.status(),
        "jobs": manager.jobs_payload(limit=limit),
        "metrics": await asyncio.to_thread(
            _tail_jsonl, settings.metrics_path, limit=20
        ),
    }


# ----------------------------------------------------------------- launcher


def _import_vllm_server() -> tuple[Any, Any, Any, Any]:
    """Import vLLM's server machinery across the entrypoints restructure.

    Returns ``(entry, cli_args, FlexibleArgumentParser, legacy_api_server)``
    where ``entry`` owns ``run_server`` and ``build_app``; ``legacy_api_server``
    is the old module carrying the module-level router, or None on new vLLM.
    """
    try:
        from vllm.entrypoints.launchers import cli_args
        from vllm.entrypoints.launchers.api_server import entry
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        return entry, cli_args, FlexibleArgumentParser, None
    except ImportError:
        import vllm.entrypoints.openai.api_server as legacy
        from vllm.entrypoints.openai import cli_args

        try:
            from vllm.utils.argparse_utils import FlexibleArgumentParser
        except ImportError:
            from vllm.utils import FlexibleArgumentParser

        return legacy, cli_args, FlexibleArgumentParser, legacy


def _install_train_routes(entry: Any, legacy: Any) -> None:
    if legacy is not None:
        # Old layout: build_app() includes the module-level router, so routes
        # added to it before the server starts ride along.
        legacy.router.include_router(train_router)
        return
    # New layout: no module-level router. build_and_serve resolves build_app
    # from the entry module's globals at call time, so wrapping it there is
    # honored.
    original_build_app = entry.build_app

    def build_app_with_training(args: Any, *extra: Any, **kwargs: Any) -> Any:
        app = original_build_app(args, *extra, **kwargs)
        app.include_router(train_router)
        return app

    entry.build_app = build_app_with_training


def _build_vllm_argv() -> list[str]:
    """Compose the vllm serve argv: our defaults first, then env extra args,
    then this process's CLI args — so explicit user flags always win."""
    argv = [
        "--host",
        settings.host,
        "--port",
        str(settings.port),
        "--gpu-memory-utilization",
        str(settings.gpu_memory_utilization),
        "--served-model-name",
        settings.served_model_name,
        "--max-loras",
        str(settings.max_loras),
        "--max-lora-rank",
        str(settings.max_lora_rank),
    ]
    if settings.tensor_parallel_size > 1:
        argv.extend(
            ["--tensor-parallel-size", str(settings.tensor_parallel_size)]
        )
    if settings.trust_remote_code:
        argv.append("--trust-remote-code")
    if settings.max_model_len:
        argv.extend(["--max-model-len", str(settings.max_model_len)])
    if settings.max_num_seqs:
        argv.extend(["--max-num-seqs", str(settings.max_num_seqs)])
    if settings.vllm_extra_args:
        argv.extend(shlex.split(settings.vllm_extra_args))
    argv.extend(sys.argv[1:])
    return argv


def _parse_vllm_args(cli_args: Any, parser_cls: Any) -> Any:
    parser = cli_args.make_arg_parser(
        parser_cls(description="vLLM inference + colocated SFT LoRA training")
    )
    argv = _build_vllm_argv()
    args = parser.parse_args(argv)
    model_flag_given = any(
        part == "--model" or part.startswith("--model=") for part in argv
    )
    if getattr(args, "model_tag", None):
        args.model = args.model_tag
    elif not model_flag_given:
        # No model on the command line: use the env setting instead of
        # vLLM's placeholder default.
        args.model = settings.model_name
    if getattr(args, "headless", False):
        raise SystemExit(
            "--headless runs no API server, so the /train/sft route cannot "
            "exist; remove the flag."
        )
    # The training route, the in-flight gate, and runtime LoRA updates all
    # need the one in-process API server.
    if getattr(args, "api_server_count", None) not in (None, 1):
        logger.warning("Forcing --api-server-count 1 (was %s)", args.api_server_count)
        args.api_server_count = 1
    args.enable_lora = True
    middleware = list(getattr(args, "middleware", None) or [])
    middleware.append(f"{__name__}.InFlightGate")
    args.middleware = middleware
    validate = getattr(cli_args, "validate_parsed_serve_args", None)
    if validate is not None:
        validate(args)
    _reconcile_network_settings(args)
    return args


def _reconcile_network_settings(args: Any) -> None:
    """The user may override --host/--port through CLI passthrough or extra
    args; the self-calls (health poll, adapter loads) must follow."""
    global settings
    host = getattr(args, "host", None) or settings.host
    port = int(getattr(args, "port", None) or settings.port)
    if (host, port) != (settings.host, settings.port):
        settings = replace(settings, host=host, port=port)
        manager.settings = settings


async def _startup() -> None:
    """Runs on the server's event loop: start the manager, then restore the
    last trained adapter once vLLM answers /health."""
    manager.start()
    if not settings.sync_on_startup:
        return
    if not await manager.wait_server_ready(settings.ready_timeout_seconds):
        logger.warning("vLLM never became ready; skipping adapter restore")
        return
    if manager.resolve_adapter_path() is not None:
        await manager.sync_adapter()


async def _serve(entry: Any, args: Any) -> None:
    startup = asyncio.get_running_loop().create_task(_startup())
    try:
        await entry.run_server(args)
    finally:
        startup.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await startup
        await manager.stop()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("VLLM_COLOCATE_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # A pause file left behind by a previous run would stall the trainer.
    with contextlib.suppress(OSError):
        settings.pause_file.unlink(missing_ok=True)
    if not settings.gpus:
        logger.warning(
            "No GPUs detected; starting anyway (vLLM will decide device placement)"
        )
    entry, cli_args, parser_cls, legacy = _import_vllm_server()
    _install_train_routes(entry, legacy)
    args = _parse_vllm_args(cli_args, parser_cls)
    logger.info(
        "Serving %s on %s:%s (GPUs [%s] shared between inference and training)",
        args.model,
        settings.host,
        settings.port,
        ",".join(settings.gpus) or "-",
    )
    try:
        import uvloop

        runner = uvloop.run
    except ImportError:
        runner = asyncio.run
    try:
        runner(_serve(entry, args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
