from __future__ import annotations

import asyncio
from collections import deque
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from .hub_checkpoints import (
    config_from_env as hub_checkpoint_config_from_env,
    download_newer_checkpoint as hub_download_newer_checkpoint,
    status_payload as hub_checkpoint_status_payload,
    upload_latest_checkpoint as hub_upload_latest_checkpoint,
)
from .paths import (
    DraftModelResolution,
    checkpoint_dir_from_env,
    native_jetspec_draft_model_resolution_from_env,
)
from .suffix_cache import CpuSuffixCache, SuffixCacheConfig


HOP_BY_HOP_HEADERS = {
    "connection",
    "content-encoding",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
INFERENCE_PATHS = {
    "v1/chat/completions",
    "v1/completions",
    "v1/responses",
}
CHAT_TEMPLATE_REQUEST_KEYS = ("chat_template", "chat_template_kwargs")
MISTRAL_ROLE_ALIASES = {"developer": "system"}
MISTRAL_TEXT_CONTENT_TYPES = {"input_text", "text"}
SPECULATIVE_METRIC_KEYWORDS = ("spec", "draft", "accept", "dflash", "jetspec", "suffix")
SUPERVISOR_ENV_PREFIXES = ("VLLM_DFLASH_", "VLLM_JETSPEC_", "VLLM_SUFFIX_")
AUTO_SPEC_METHODS = {"auto", "best", "fast", "performance"}
CUSTOM_HYBRID_SPEC_METHODS = {
    "custom_class",
    "hybrid",
    "hybrid_plugin",
    "plugin",
    "plugin_hybrid",
}
JETSPEC_SPEC_METHODS = {"jetspec", "jet_spec", "tree", "parallel_tree"}
VLLM_IMAGE_METADATA_ENV = {
    "VLLM_BUILD_COMMIT",
    "VLLM_BUILD_PIPELINE",
    "VLLM_BUILD_URL",
    "VLLM_IMAGE_TAG",
}
CUSTOM_PROPOSER_ENV = {
    "VLLM_JETSPEC_MODEL",
    "VLLM_JETSPEC_DATA_DIR",
    "VLLM_JETSPEC_SUFFIX_CACHE_PATH",
    "VLLM_JETSPEC_CHECKPOINT_DIR",
    "VLLM_JETSPEC_CHECKPOINT_ROOT",
    "VLLM_JETSPEC_LATEST_CHECKPOINT_JSON",
    "VLLM_JETSPEC_ADAPTER_CHECKPOINT",
    "VLLM_JETSPEC_PROPOSER_DEVICE",
    "VLLM_JETSPEC_PROPOSER_STATS_PATH",
    "VLLM_JETSPEC_PROPOSER_STATS_FLUSH_EVERY",
    "VLLM_JETSPEC_PROPOSER_STATS_FLUSH_SECONDS",
    "VLLM_JETSPEC_MAX_PROPOSED_TOKENS",
    "VLLM_JETSPEC_MAX_ADAPTER_TOKENS",
    "VLLM_JETSPEC_MIN_ADAPTER_TOKENS",
    "VLLM_JETSPEC_FORCE_HYBRID_PROPOSALS",
    "VLLM_JETSPEC_MIN_TOKEN_PROB",
    "VLLM_JETSPEC_SUFFIX_BACKEND",
    "VLLM_JETSPEC_SUFFIX_RELOAD_SECONDS",
    "VLLM_JETSPEC_ADAPTER_RELOAD_SECONDS",
    "VLLM_JETSPEC_IN_CONTEXT_SUFFIX",
    "VLLM_JETSPEC_IN_CONTEXT_SUFFIX_BOOTSTRAP",
    "VLLM_JETSPEC_IN_CONTEXT_SUFFIX_WINDOW",
    "VLLM_JETSPEC_IN_CONTEXT_SUFFIX_MIN_MATCH",
    "VLLM_SUFFIX_MAX_TREE_DEPTH",
    "VLLM_SUFFIX_MAX_CACHED_REQUESTS",
    "VLLM_SUFFIX_MAX_SPEC_FACTOR",
    "VLLM_SUFFIX_MIN_TOKEN_PROB",
    "VLLM_SUFFIX_MIN_MATCH_TOKENS",
}
CUSTOM_PROPOSER_ENV_PREFIX_ALIASES = (
    ("VLLM_JETSPEC_", "JETSPEC_PLUGIN_"),
    ("VLLM_SUFFIX_", "JETSPEC_SUFFIX_"),
)
PROMETHEUS_SAMPLE_RE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)$"
)
logger = logging.getLogger("vllm_dflash_jit.app")


def _custom_proposer_env_name(name: str) -> str:
    for prefix, alias_prefix in CUSTOM_PROPOSER_ENV_PREFIX_ALIASES:
        if name.startswith(prefix):
            return f"{alias_prefix}{name[len(prefix):]}"
    return name


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw != "":
            return raw
    return default


def _env_first_str(*names: str, default: str) -> str:
    value = _env_first(*names, default=default)
    return str(value if value is not None else default)


def _env_bool_any(names: tuple[str, ...], default: bool) -> bool:
    for name in names:
        raw = os.getenv(name)
        if raw is not None:
            return raw.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _bool_cli_option(name: str, enabled: bool) -> list[str]:
    return [name if enabled else f"--no-{name[2:]}"]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_int_any(names: tuple[str, ...], default: int) -> int:
    raw = _env_first(*names)
    if raw is None or raw == "":
        return default
    return int(raw)


def _safe_int(value: Any) -> int | None:
    changed = False
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_float_any(names: tuple[str, ...], default: float) -> float:
    raw = _env_first(*names)
    if raw is None or raw == "":
        return default
    return float(raw)


def _safe_runtime_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    if cleaned:
        return cleaned[:96]
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _default_ephemeral_data_dir(model_name: str) -> Path:
    return Path(
        _env_first_str(
            "VLLM_JETSPEC_EPHEMERAL_DATA_DIR",
            "VLLM_DFLASH_EPHEMERAL_DATA_DIR",
            default=f"/dev/shm/vllm_jetspec/{_safe_runtime_slug(model_name)}",
        )
    )


def _is_ephemeral_runtime_path(path: Path) -> bool:
    raw = str(path)
    return raw == "/dev/shm" or raw.startswith(("/dev/shm/", "/run/shm/", "/tmp/"))


def _runtime_data_path(
    names: tuple[str, ...],
    *,
    persist: bool,
    persistent_default: Path,
    ephemeral_dir: Path,
    ephemeral_filename: str,
) -> Path:
    raw = _env_first(*names)
    if persist:
        return Path(raw) if raw else persistent_default
    if raw:
        requested = Path(raw)
        if _is_ephemeral_runtime_path(requested):
            return requested
        return ephemeral_dir / requested.name
    return ephemeral_dir / ephemeral_filename


def _vllm_parallel_sizes(extra_args: str) -> dict[str, int]:
    values = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
    }
    try:
        parts = shlex.split(str(extra_args or ""))
    except ValueError:
        return values
    names = {
        "--tensor-parallel-size": "tensor_parallel_size",
        "--pipeline-parallel-size": "pipeline_parallel_size",
        "--data-parallel-size": "data_parallel_size",
    }
    index = 0
    while index < len(parts):
        part = parts[index]
        key = None
        raw_value = None
        if "=" in part:
            option, raw_value = part.split("=", 1)
            key = names.get(option)
        else:
            key = names.get(part)
            if key and index + 1 < len(parts):
                raw_value = parts[index + 1]
                index += 1
        if key and raw_value is not None:
            parsed = _safe_int(raw_value)
            if parsed is not None and parsed > 0:
                values[key] = parsed
        index += 1
    return values


def _vllm_extra_arg_values(extra_args: str, names: set[str]) -> list[str]:
    try:
        parts = shlex.split(str(extra_args or ""))
    except ValueError:
        return []

    values: list[str] = []
    index = 0
    while index < len(parts):
        part = parts[index]
        option, has_inline_value, raw_value = part.partition("=")
        if option in names:
            if has_inline_value:
                values.append(raw_value)
            elif index + 1 < len(parts):
                values.append(parts[index + 1])
                index += 1
        index += 1
    return values


def _vllm_uses_mistral_tokenizer(settings: Any | None) -> bool:
    extra_args = getattr(settings, "vllm_extra_args", "")
    mistral_options = {
        "--tokenizer_mode",
        "--tokenizer-mode",
        "--config_format",
        "--config-format",
        "--load_format",
        "--load-format",
    }
    return any(
        value.strip().lower() == "mistral"
        for value in _vllm_extra_arg_values(extra_args, mistral_options)
    )


def _serving_model_parallel_size(settings: Any) -> int:
    sizes = _vllm_parallel_sizes(getattr(settings, "vllm_extra_args", ""))
    return max(1, sizes["tensor_parallel_size"]) * max(
        1, sizes["pipeline_parallel_size"]
    )


def _extra_arg_present(extra_args: str, option: str) -> bool:
    try:
        parts = shlex.split(extra_args or "")
    except ValueError:
        return False
    return option in parts or any(part.startswith(f"{option}=") for part in parts)


def _live_lora_serving_enabled(settings: Any) -> bool:
    return bool(getattr(settings, "live_lora_serving_enabled", False))


def _live_lora_proxy_enabled() -> bool:
    return _env_bool_any(
        (
            "VLLM_JETSPEC_LIVE_LORA_PROXY_AUTO",
            "VLLM_DFLASH_LIVE_LORA_PROXY_AUTO",
        ),
        True,
    )


def _live_trainer_parallel_mode(settings: Any) -> str:
    mode = (
        _env_first_str(
            "VLLM_JETSPEC_LIVE_TRAIN_PARALLEL_MODE",
            "VLLM_DFLASH_LIVE_TRAIN_PARALLEL_MODE",
            default="auto",
        )
        .strip()
        .lower()
        .replace("-", "_")
    )
    aliases = {
        "off": "none",
        "false": "none",
        "single": "none",
        "single_gpu": "none",
        "device": "device_map",
        "model": "model_parallel",
        "mp": "model_parallel",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"auto", "none", "device_map", "model_parallel"}:
        mode = "auto"
    if mode == "auto" and _serving_model_parallel_size(settings) > 1:
        return "model_parallel"
    return mode


def _visible_cuda_device_count() -> int:
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            continue
        normalized = raw.strip().lower()
        if normalized in {"all", "void", "none"}:
            return 0
        return len([part for part in raw.split(",") if part.strip()])
    return 0


def _trainer_parallel_mode(settings: Any) -> str:
    mode = (
        _env_first_str(
            "VLLM_JETSPEC_TRAIN_PARALLEL_MODE",
            "VLLM_DFLASH_TRAIN_PARALLEL_MODE",
            default="auto",
        )
        .strip()
        .lower()
        .replace("-", "_")
    )
    aliases = {
        "off": "none",
        "false": "none",
        "single": "none",
        "single_gpu": "none",
        "data": "ddp",
        "data_parallel": "ddp",
        "distributed": "ddp",
        "distributed_data_parallel": "ddp",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"auto", "none", "ddp"}:
        mode = "auto"
    if mode == "auto" and _serving_model_parallel_size(settings) > 1:
        return "ddp"
    return mode


def _trainer_parallel_world_size(settings: Any) -> int:
    explicit = _optional_int_any(
        ("VLLM_JETSPEC_TRAIN_WORLD_SIZE", "VLLM_DFLASH_TRAIN_WORLD_SIZE"), None
    )
    if explicit is not None:
        return max(1, explicit)
    if _trainer_parallel_mode(settings) == "none":
        return 1
    served_parallel = _serving_model_parallel_size(settings)
    if served_parallel <= 1:
        return 1
    visible_devices = _visible_cuda_device_count()
    if visible_devices > 0:
        return max(1, min(served_parallel, visible_devices))
    return served_parallel


def _trainer_failure_backoff_seconds(failure_count: int) -> float:
    base = _env_float_any(
        (
            "VLLM_JETSPEC_TRAIN_FAILURE_BACKOFF_SECONDS",
            "VLLM_DFLASH_TRAIN_FAILURE_BACKOFF_SECONDS",
        ),
        60.0,
    )
    max_delay = _env_float_any(
        (
            "VLLM_JETSPEC_TRAIN_FAILURE_BACKOFF_MAX_SECONDS",
            "VLLM_DFLASH_TRAIN_FAILURE_BACKOFF_MAX_SECONDS",
        ),
        900.0,
    )
    return min(max_delay, max(0.0, base) * (2 ** min(max(0, failure_count - 1), 5)))


@dataclass(frozen=True)
class Settings:
    model_name: str
    served_model_name: str
    public_host: str
    public_port: int
    vllm_host: str
    vllm_port: int
    data_dir: Path
    ephemeral_data_dir: Path
    traffic_path: Path
    training_data_path: Path
    live_training_data_path: Path
    suffix_cache_path: Path
    persist_traffic: bool
    persist_suffix_cache: bool
    checkpoint_dir: Path
    metrics_path: Path
    proposer_stats_path: Path
    stop_file: Path
    train_steps_per_idle: int
    train_log_every: int
    min_training_examples: int
    min_live_training_examples: int
    training_enabled: bool
    live_training_enabled: bool
    live_lora_serving_enabled: bool
    live_lora_checkpoint_dir: Path
    live_lora_adapter_name: str
    wake_after_training: bool
    sleep_level: int
    gpu_memory_utilization: float
    max_model_len: int | None
    max_num_seqs: int | None
    trust_remote_code: bool
    enable_sleep_mode: bool
    hf_overrides: dict[str, Any]
    vllm_extra_args: str
    spec_method: str
    speculative_config: dict[str, Any]
    jetspec_draft_model_resolution: DraftModelResolution | None
    trainer_command: str

    @property
    def vllm_base_url(self) -> str:
        return f"http://{self.vllm_host}:{self.vllm_port}"

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(
            _env_first_str(
                "VLLM_JETSPEC_DATA_DIR",
                "VLLM_DFLASH_DATA_DIR",
                default="/data/vllm_dflash",
            )
        )
        model_name = _env_first_str(
            "VLLM_JETSPEC_MODEL", "VLLM_DFLASH_MODEL", default="Qwen/Qwen3-8B"
        )
        ephemeral_data_dir = _default_ephemeral_data_dir(model_name)
        persist_traffic = _env_bool_any(
            ("VLLM_JETSPEC_PERSIST_TRAFFIC", "VLLM_DFLASH_PERSIST_TRAFFIC"),
            False,
        )
        persist_suffix_cache = _env_bool_any(
            (
                "VLLM_JETSPEC_PERSIST_SUFFIX_CACHE",
                "VLLM_DFLASH_PERSIST_SUFFIX_CACHE",
            ),
            False,
        )
        traffic_path = _runtime_data_path(
            ("VLLM_JETSPEC_TRAFFIC_PATH", "VLLM_DFLASH_TRAFFIC_PATH"),
            persist=persist_traffic,
            persistent_default=data_dir / "traffic.jsonl",
            ephemeral_dir=ephemeral_data_dir,
            ephemeral_filename="traffic.jsonl",
        )
        training_data_path = _runtime_data_path(
            (
                "VLLM_JETSPEC_TRAIN_DATA_PATH",
                "VLLM_JETSPEC_DATASET_PATH",
                "VLLM_DFLASH_TRAIN_DATA_PATH",
                "VLLM_DFLASH_DATASET_PATH",
            ),
            persist=persist_traffic,
            persistent_default=traffic_path,
            ephemeral_dir=ephemeral_data_dir,
            ephemeral_filename="traffic.jsonl",
        )
        live_training_data_path = Path(
            _env_first_str(
                "VLLM_JETSPEC_LIVE_TRAIN_DATA_PATH",
                "VLLM_DFLASH_LIVE_TRAIN_DATA_PATH",
                default=str(data_dir / "live_training.jsonl"),
            )
        )
        suffix_cache_path = _runtime_data_path(
            ("VLLM_JETSPEC_SUFFIX_CACHE_PATH", "VLLM_DFLASH_SUFFIX_CACHE_PATH"),
            persist=persist_suffix_cache,
            persistent_default=data_dir / "suffix_cache.json",
            ephemeral_dir=ephemeral_data_dir,
            ephemeral_filename="suffix_cache.json",
        )
        spec_method = _spec_method_from_env()
        jetspec_draft_model_resolution = native_jetspec_draft_model_resolution_from_env(
            model_name, data_dir
        )
        speculative_config = _load_speculative_config(
            model_name=model_name,
            data_dir=data_dir,
            spec_method=spec_method,
            jetspec_draft_model_resolution=jetspec_draft_model_resolution,
        )
        return cls(
            model_name=model_name,
            served_model_name=_env_first_str(
                "VLLM_JETSPEC_SERVED_MODEL_NAME",
                "VLLM_DFLASH_SERVED_MODEL_NAME",
                default="smolagent-dflash",
            ),
            public_host=_env_first_str(
                "VLLM_JETSPEC_PUBLIC_HOST", "VLLM_DFLASH_PUBLIC_HOST", default="0.0.0.0"
            ),
            public_port=_env_int_any(
                ("VLLM_JETSPEC_PUBLIC_PORT", "VLLM_DFLASH_PUBLIC_PORT"), 30006
            ),
            vllm_host=_env_first_str(
                "VLLM_JETSPEC_INTERNAL_HOST",
                "VLLM_DFLASH_INTERNAL_HOST",
                default="127.0.0.1",
            ),
            vllm_port=_env_int_any(
                ("VLLM_JETSPEC_INTERNAL_PORT", "VLLM_DFLASH_INTERNAL_PORT"), 30007
            ),
            data_dir=data_dir,
            ephemeral_data_dir=ephemeral_data_dir,
            traffic_path=traffic_path,
            training_data_path=training_data_path,
            live_training_data_path=live_training_data_path,
            suffix_cache_path=suffix_cache_path,
            persist_traffic=persist_traffic,
            persist_suffix_cache=persist_suffix_cache,
            checkpoint_dir=checkpoint_dir_from_env(model_name, data_dir),
            metrics_path=Path(
                _env_first_str(
                    "VLLM_JETSPEC_METRICS_PATH",
                    "VLLM_DFLASH_METRICS_PATH",
                    default=str(ephemeral_data_dir / "trainer_metrics.jsonl"),
                )
            ),
            proposer_stats_path=Path(
                _env_first_str(
                    "VLLM_JETSPEC_PROPOSER_STATS_PATH",
                    "VLLM_DFLASH_PROPOSER_STATS_PATH",
                    default=str(ephemeral_data_dir / "proposer_stats.json"),
                )
            ),
            stop_file=Path(
                _env_first_str(
                    "VLLM_JETSPEC_STOP_FILE",
                    "VLLM_DFLASH_STOP_FILE",
                    default=str(ephemeral_data_dir / "stop_training"),
                )
            ),
            train_steps_per_idle=_env_int_any(
                ("VLLM_JETSPEC_TRAIN_STEPS", "VLLM_DFLASH_TRAIN_STEPS"), 0
            ),
            train_log_every=_env_int_any(
                ("VLLM_JETSPEC_TRAIN_LOG_EVERY", "VLLM_DFLASH_TRAIN_LOG_EVERY"), 1
            ),
            min_training_examples=_env_int_any(
                ("VLLM_JETSPEC_MIN_EXAMPLES", "VLLM_DFLASH_MIN_EXAMPLES"), 8
            ),
            min_live_training_examples=_env_int_any(
                (
                    "VLLM_JETSPEC_LIVE_MIN_EXAMPLES",
                    "VLLM_DFLASH_LIVE_MIN_EXAMPLES",
                ),
                1,
            ),
            training_enabled=_env_bool_any(
                ("VLLM_JETSPEC_TRAINING_ENABLED", "VLLM_DFLASH_TRAINING_ENABLED"),
                False,
            ),
            live_training_enabled=_env_bool_any(
                (
                    "VLLM_JETSPEC_LIVE_TRAINING_ENABLED",
                    "VLLM_DFLASH_LIVE_TRAINING_ENABLED",
                ),
                True,
            ),
            live_lora_serving_enabled=_env_bool_any(
                (
                    "VLLM_JETSPEC_LIVE_LORA_SERVING_ENABLED",
                    "VLLM_DFLASH_LIVE_LORA_SERVING_ENABLED",
                ),
                False,
            ),
            live_lora_checkpoint_dir=checkpoint_dir_from_env(model_name, data_dir)
            / "live_lora",
            live_lora_adapter_name=_env_first_str(
                "VLLM_JETSPEC_LIVE_LORA_ADAPTER_NAME",
                "VLLM_DFLASH_LIVE_LORA_ADAPTER_NAME",
                default=f"{_safe_runtime_slug(model_name)}-live-lora",
            ),
            wake_after_training=_env_bool_any(
                ("VLLM_JETSPEC_WAKE_AFTER_TRAINING", "VLLM_DFLASH_WAKE_AFTER_TRAINING"),
                False,
            ),
            sleep_level=_env_int_any(
                ("VLLM_JETSPEC_SLEEP_LEVEL", "VLLM_DFLASH_SLEEP_LEVEL"), 1
            ),
            gpu_memory_utilization=_env_float_any(
                (
                    "VLLM_JETSPEC_GPU_MEMORY_UTILIZATION",
                    "VLLM_DFLASH_GPU_MEMORY_UTILIZATION",
                ),
                0.72,
            ),
            max_model_len=_optional_int_any(
                ("VLLM_JETSPEC_MAX_MODEL_LEN", "VLLM_DFLASH_MAX_MODEL_LEN"), 32768
            ),
            max_num_seqs=_optional_int_any(
                ("VLLM_JETSPEC_MAX_NUM_SEQS", "VLLM_DFLASH_MAX_NUM_SEQS"), 16
            ),
            trust_remote_code=_env_bool_any(
                ("VLLM_JETSPEC_TRUST_REMOTE_CODE", "VLLM_DFLASH_TRUST_REMOTE_CODE"),
                True,
            ),
            enable_sleep_mode=_env_bool_any(
                ("VLLM_JETSPEC_ENABLE_SLEEP_MODE", "VLLM_DFLASH_ENABLE_SLEEP_MODE"),
                True,
            ),
            hf_overrides=_env_json_obj_any(
                ("VLLM_JETSPEC_HF_OVERRIDES", "VLLM_DFLASH_HF_OVERRIDES")
            ),
            vllm_extra_args=_env_first_str(
                "VLLM_JETSPEC_VLLM_EXTRA_ARGS",
                "VLLM_DFLASH_VLLM_EXTRA_ARGS",
                default="",
            ),
            spec_method=spec_method,
            speculative_config=speculative_config,
            jetspec_draft_model_resolution=jetspec_draft_model_resolution,
            trainer_command=_env_first_str(
                "VLLM_JETSPEC_TRAINER_COMMAND",
                "VLLM_DFLASH_TRAINER_COMMAND",
                default="",
            ),
        )


def _optional_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    if raw == "":
        return None
    return int(raw)


def _optional_int_any(names: tuple[str, ...], default: int | None) -> int | None:
    for name in names:
        raw = os.getenv(name)
        if raw is not None:
            if raw == "":
                return None
            return int(raw)
    return default


def _env_json_obj(name: str) -> dict[str, Any]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _env_json_obj_any(names: tuple[str, ...]) -> dict[str, Any]:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw.strip() != "":
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a JSON object")
            return value
    return {}


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in re.split(r"[\s,]+", raw.strip()):
        if not part:
            continue
        values.append(int(part))
    return values


def _parse_compile_sizes(raw: str) -> list[int | str]:
    values: list[int | str] = []
    for part in re.split(r"[\s,]+", raw.strip()):
        if not part:
            continue
        parsed = _safe_int(part)
        values.append(parsed if parsed is not None else part)
    return values


def _vllm_compilation_config_from_env() -> str | dict[str, Any] | None:
    raw_config = _env_first(
        "VLLM_JETSPEC_VLLM_COMPILATION_CONFIG",
        "VLLM_DFLASH_VLLM_COMPILATION_CONFIG",
    )
    if raw_config:
        return raw_config

    mode = _env_first(
        "VLLM_JETSPEC_VLLM_COMPILE_MODE",
        "VLLM_DFLASH_VLLM_COMPILE_MODE",
    )
    cudagraph_mode = _env_first(
        "VLLM_JETSPEC_VLLM_CUDAGRAPH_MODE",
        "VLLM_DFLASH_VLLM_CUDAGRAPH_MODE",
    )
    cudagraph_sizes = _env_first(
        "VLLM_JETSPEC_VLLM_CUDAGRAPH_CAPTURE_SIZES",
        "VLLM_DFLASH_VLLM_CUDAGRAPH_CAPTURE_SIZES",
    )
    compile_sizes = _env_first(
        "VLLM_JETSPEC_VLLM_COMPILE_SIZES",
        "VLLM_DFLASH_VLLM_COMPILE_SIZES",
    )
    max_cudagraph_capture_size = _optional_int_any(
        (
            "VLLM_JETSPEC_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE",
            "VLLM_DFLASH_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE",
        ),
        None,
    )
    if (
        not mode
        and not cudagraph_mode
        and not cudagraph_sizes
        and not compile_sizes
        and max_cudagraph_capture_size is None
    ):
        return None

    config: dict[str, Any] = {}
    if mode:
        parsed_mode = _safe_int(mode)
        config["mode"] = parsed_mode if parsed_mode is not None else mode
    if cudagraph_mode:
        config["cudagraph_mode"] = cudagraph_mode
    if cudagraph_sizes:
        config["cudagraph_capture_sizes"] = _parse_int_list(cudagraph_sizes)
    if compile_sizes:
        config["compile_sizes"] = _parse_compile_sizes(compile_sizes)
    if max_cudagraph_capture_size is not None:
        config["max_cudagraph_capture_size"] = max_cudagraph_capture_size
    return config


def _spec_method_from_env() -> str:
    return (
        _env_first_str(
            "VLLM_JETSPEC_SPEC_METHOD", "VLLM_DFLASH_SPEC_METHOD", default="auto"
        )
        .strip()
        .lower()
    )


def _custom_hybrid_speculative_config() -> dict[str, Any]:
    return {
        "method": "custom_class",
        "model": "vllm_dflash_jit.hybrid_proposer.HybridSuffixJetSpecProposer",
        "num_speculative_tokens": _env_int_any(
            (
                "VLLM_JETSPEC_NUM_SPECULATIVE_TOKENS",
                "VLLM_DFLASH_NUM_SPECULATIVE_TOKENS",
            ),
            8,
        ),
    }


def _suffix_speculative_config() -> dict[str, Any]:
    return {
        "method": "suffix",
        "num_speculative_tokens": _env_int_any(
            (
                "VLLM_JETSPEC_NUM_SPECULATIVE_TOKENS",
                "VLLM_DFLASH_NUM_SPECULATIVE_TOKENS",
            ),
            16,
        ),
        "suffix_decoding_max_tree_depth": _env_int("VLLM_SUFFIX_MAX_TREE_DEPTH", 24),
        "suffix_decoding_max_cached_requests": _env_int(
            "VLLM_SUFFIX_MAX_CACHED_REQUESTS", 10_000
        ),
        "suffix_decoding_max_spec_factor": _env_float(
            "VLLM_SUFFIX_MAX_SPEC_FACTOR", 1.0
        ),
        "suffix_decoding_min_token_prob": _env_float("VLLM_SUFFIX_MIN_TOKEN_PROB", 0.1),
    }


def _auto_spec_method(
    *,
    jetspec_draft_model_resolution: DraftModelResolution | None,
) -> str:
    policy = _env_first_str(
        "VLLM_JETSPEC_AUTO_POLICY",
        "VLLM_DFLASH_AUTO_POLICY",
        default="plugin_hybrid",
    ).strip().lower()
    if policy in {"suffix", "native_suffix"}:
        return "suffix"
    if policy in {"plugin", "plugin_hybrid", "hybrid"}:
        return "custom_class"
    if policy in JETSPEC_SPEC_METHODS and (
        jetspec_draft_model_resolution is not None
        and bool(jetspec_draft_model_resolution.model)
    ):
        return "jetspec"
    return "custom_class"


def _load_speculative_config(
    *,
    model_name: str | None = None,
    data_dir: str | os.PathLike[str] | None = None,
    spec_method: str | None = None,
    jetspec_draft_model_resolution: DraftModelResolution | None = None,
) -> dict[str, Any]:
    raw = _env_first(
        "VLLM_JETSPEC_SPECULATIVE_CONFIG", "VLLM_DFLASH_SPECULATIVE_CONFIG"
    )
    if raw:
        return json.loads(raw)
    method = (spec_method or _spec_method_from_env()).strip().lower()
    if method in {"none", "off", "disabled", "false", "0"}:
        return {}
    if method in AUTO_SPEC_METHODS:
        if jetspec_draft_model_resolution is None:
            jetspec_draft_model_resolution = (
                native_jetspec_draft_model_resolution_from_env(
                    model_name
                    or _env_first_str(
                        "VLLM_JETSPEC_MODEL",
                        "VLLM_DFLASH_MODEL",
                        default="Qwen/Qwen3-8B",
                    ),
                    data_dir
                    or _env_first_str(
                        "VLLM_JETSPEC_DATA_DIR",
                        "VLLM_DFLASH_DATA_DIR",
                        default="/data/vllm_dflash",
                    ),
                )
            )
        method = _auto_spec_method(
            jetspec_draft_model_resolution=jetspec_draft_model_resolution
        )
    if method in CUSTOM_HYBRID_SPEC_METHODS:
        return _custom_hybrid_speculative_config()
    if method in JETSPEC_SPEC_METHODS:
        if jetspec_draft_model_resolution is None:
            jetspec_draft_model_resolution = (
                native_jetspec_draft_model_resolution_from_env(
                    model_name
                    or _env_first_str(
                        "VLLM_JETSPEC_MODEL",
                        "VLLM_DFLASH_MODEL",
                        default="Qwen/Qwen3-8B",
                    ),
                    data_dir
                    or _env_first_str(
                        "VLLM_JETSPEC_DATA_DIR",
                        "VLLM_DFLASH_DATA_DIR",
                        default="/data/vllm_dflash",
                    ),
                )
            )
        if not jetspec_draft_model_resolution.model:
            raise ValueError(
                "VLLM_JETSPEC_SPEC_METHOD=jetspec requires a compatible JetSpec "
                "draft head. Set VLLM_JETSPEC_DRAFT_HEAD/VLLM_JETSPEC_DRAFT_MODEL, "
                "point VLLM_JETSPEC_DRAFT_HEAD_DIR at a local Hugging Face draft-head "
                "directory, or place a draft head under the model-scoped "
                "checkpoints/<model>/jetspec folder."
            )
        config = {
            "method": "dflash",
            "model": jetspec_draft_model_resolution.model,
            "num_speculative_tokens": _env_int_any(
                (
                    "VLLM_JETSPEC_NUM_SPECULATIVE_TOKENS",
                    "VLLM_DFLASH_NUM_SPECULATIVE_TOKENS",
                ),
                15,
            ),
            "max_model_len": _optional_int_any(
                ("VLLM_JETSPEC_DRAFT_MAX_MODEL_LEN", "VLLM_DFLASH_DRAFT_MAX_MODEL_LEN"),
                None,
            ),
            "head_type": _env_first_str(
                "VLLM_JETSPEC_HEAD_TYPE", "VLLM_DFLASH_HEAD_TYPE", default="causal"
            ),
            "tree_width": _env_int_any(
                ("VLLM_JETSPEC_TREE_WIDTH", "VLLM_DFLASH_TREE_WIDTH"), 7
            ),
            "max_tree_budget": _optional_int_any(
                ("VLLM_JETSPEC_MAX_TREE_BUDGET", "VLLM_DFLASH_MAX_TREE_BUDGET"),
                _env_int_any(
                    ("VLLM_JETSPEC_TREE_BUDGET", "VLLM_DFLASH_TREE_BUDGET"), 128
                ),
            ),
            "tree_draft": _env_first_str(
                "VLLM_JETSPEC_TREE_DRAFT",
                "VLLM_DFLASH_TREE_DRAFT",
                default="accum_logp",
            ),
            "tree_hybrid_alpha": _env_float_any(
                ("VLLM_JETSPEC_TREE_HYBRID_ALPHA", "VLLM_DFLASH_TREE_HYBRID_ALPHA"),
                1.0,
            ),
            "max_draft_passes": _env_int_any(
                ("VLLM_JETSPEC_MAX_DRAFT_PASSES", "VLLM_DFLASH_MAX_DRAFT_PASSES"),
                0,
            ),
            "tree_prune_ratio": _env_float_any(
                ("VLLM_JETSPEC_TREE_PRUNE_RATIO", "VLLM_DFLASH_TREE_PRUNE_RATIO"),
                0.25,
            ),
            "tree_construction": _env_first_str(
                "VLLM_JETSPEC_TREE_CONSTRUCTION",
                "VLLM_DFLASH_TREE_CONSTRUCTION",
                default="breadth_first",
            ),
            "tree_attn_kernel": _env_first_str(
                "VLLM_JETSPEC_TREE_ATTN_KERNEL",
                "VLLM_DFLASH_TREE_ATTN_KERNEL",
                default="triton",
            ),
            "tree_kv_layout": _env_first_str(
                "VLLM_JETSPEC_TREE_KV_LAYOUT",
                "VLLM_DFLASH_TREE_KV_LAYOUT",
                default="logical",
            ),
            "num_cudagraph_tree_captures": _env_int_any(
                (
                    "VLLM_JETSPEC_NUM_CUDAGRAPH_TREE_CAPTURES",
                    "VLLM_DFLASH_NUM_CUDAGRAPH_TREE_CAPTURES",
                ),
                0,
            ),
        }
        return {key: value for key, value in config.items() if value not in {"", None}}
    if method == "dflash":
        raise ValueError(
            "Native DFlash serving has been removed from this wrapper. Use "
            "VLLM_JETSPEC_SPEC_METHOD=jetspec for the JetSpec vLLM fork, "
            "plugin_hybrid for suffix+JetSpec plugin serving, or suffix."
        )
    return _suffix_speculative_config()


class VllmProcess:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.process: subprocess.Popen[Any] | None = None

    def status(self) -> dict[str, Any]:
        process = self.process
        if process is None:
            return {"running": False, "pid": None, "returncode": None}
        return {
            "running": process.poll() is None,
            "pid": process.pid,
            "returncode": process.returncode,
        }

    async def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._command()
        env = self._child_env()
        logger.info("Starting vLLM: %s", shlex.join(cmd))
        self.process = subprocess.Popen(cmd, env=env)
        await self.wait_ready()
        logger.info("vLLM is ready at %s", self.settings.vllm_base_url)

    def _child_env(self) -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(SUPERVISOR_ENV_PREFIXES)
        }
        for key in VLLM_IMAGE_METADATA_ENV:
            env.pop(key, None)
        env.setdefault("VLLM_SERVER_DEV_MODE", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        torch_float32_matmul_precision = (
            _env_first(
                "VLLM_JETSPEC_TORCH_FLOAT32_MATMUL_PRECISION",
                "TORCH_FLOAT32_MATMUL_PRECISION",
                default="high",
            )
            or ""
        ).strip()
        if torch_float32_matmul_precision:
            env["TORCH_FLOAT32_MATMUL_PRECISION"] = (
                torch_float32_matmul_precision
            )
        if self.settings.speculative_config.get("method") == "custom_class":
            def set_custom_proposer_env(key: str, value: Any) -> None:
                env[_custom_proposer_env_name(key)] = str(value)

            for key in CUSTOM_PROPOSER_ENV:
                alias = _custom_proposer_env_name(key)
                if alias in os.environ:
                    set_custom_proposer_env(key, os.environ[alias])
                elif key in os.environ:
                    set_custom_proposer_env(key, os.environ[key])
            custom_defaults = {
                "VLLM_JETSPEC_MODEL": self.settings.model_name,
                "VLLM_JETSPEC_DATA_DIR": str(self.settings.data_dir),
                "VLLM_JETSPEC_SUFFIX_CACHE_PATH": str(self.settings.suffix_cache_path),
                "VLLM_JETSPEC_CHECKPOINT_ROOT": str(self.settings.checkpoint_dir.parent),
                "VLLM_JETSPEC_LATEST_CHECKPOINT_JSON": str(
                    self.settings.checkpoint_dir / "latest.json"
                ),
                "VLLM_JETSPEC_ADAPTER_CHECKPOINT": str(
                    self.settings.checkpoint_dir / "latest.json"
                ),
                "VLLM_JETSPEC_PROPOSER_STATS_PATH": str(
                    self.settings.proposer_stats_path
                ),
            }
            for key, value in custom_defaults.items():
                env.setdefault(_custom_proposer_env_name(key), value)
        return env

    def _command(self) -> list[str]:
        cmd = [
            "vllm",
            "serve",
            self.settings.model_name,
            "--served-model-name",
            self.settings.served_model_name,
            "--host",
            self.settings.vllm_host,
            "--port",
            str(self.settings.vllm_port),
            "--gpu-memory-utilization",
            str(self.settings.gpu_memory_utilization),
        ]
        if self.settings.speculative_config:
            cmd.extend(
                [
                    "--speculative-config",
                    json.dumps(self.settings.speculative_config, separators=(",", ":")),
                ]
            )
            if self.settings.speculative_config.get("method") == "custom_class":
                disable_async_scheduling = _env_bool_any(
                    (
                        "VLLM_JETSPEC_DISABLE_ASYNC_SCHEDULING_FOR_CUSTOM_PROPOSER",
                        "VLLM_DFLASH_DISABLE_ASYNC_SCHEDULING_FOR_CUSTOM_PROPOSER",
                    ),
                    True,
                )
                if disable_async_scheduling:
                    cmd.append("--no-async-scheduling")
        if self.settings.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.settings.enable_sleep_mode:
            cmd.append("--enable-sleep-mode")
        if _live_lora_serving_enabled(self.settings):
            extra_args = self.settings.vllm_extra_args
            if not _extra_arg_present(extra_args, "--enable-lora"):
                cmd.append("--enable-lora")
            if not _extra_arg_present(extra_args, "--max-loras"):
                cmd.extend(
                    [
                        "--max-loras",
                        _env_first_str(
                            "VLLM_JETSPEC_LIVE_MAX_LORAS",
                            "VLLM_DFLASH_LIVE_MAX_LORAS",
                            default="1",
                        ),
                    ]
                )
            if not _extra_arg_present(extra_args, "--max-lora-rank"):
                cmd.extend(
                    [
                        "--max-lora-rank",
                        _env_first_str(
                            "VLLM_JETSPEC_LIVE_MAX_LORA_RANK",
                            "VLLM_DFLASH_LIVE_MAX_LORA_RANK",
                            default="64",
                        ),
                    ]
                )
        if self.settings.max_model_len:
            cmd.extend(["--max-model-len", str(self.settings.max_model_len)])
        if self.settings.max_num_seqs:
            cmd.extend(["--max-num-seqs", str(self.settings.max_num_seqs)])
        if self.settings.hf_overrides:
            cmd.extend(
                [
                    "--hf-overrides",
                    json.dumps(self.settings.hf_overrides, separators=(",", ":")),
                ]
            )
        compilation_config = _vllm_compilation_config_from_env()
        if compilation_config:
            cmd.extend(
                [
                    "--compilation-config",
                    (
                        compilation_config
                        if isinstance(compilation_config, str)
                        else json.dumps(compilation_config, separators=(",", ":"))
                    ),
                ]
            )
        if self.settings.vllm_extra_args:
            cmd.extend(shlex.split(self.settings.vllm_extra_args))
        return cmd

    async def wait_ready(self, timeout_seconds: float | None = None) -> None:
        if timeout_seconds is None:
            timeout_seconds = _env_float_any(
                (
                    "VLLM_JETSPEC_READY_TIMEOUT_SECONDS",
                    "VLLM_DFLASH_READY_TIMEOUT_SECONDS",
                ),
                900.0,
            )
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.monotonic() < deadline:
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError(
                        f"vLLM exited with code {self.process.returncode}"
                    )
                try:
                    response = await client.get(
                        f"{self.settings.vllm_base_url}/v1/models"
                    )
                    if response.status_code < 500:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(2)
        raise TimeoutError("Timed out waiting for vLLM to become ready")

    async def sleep(self, level: int) -> None:
        await self._post_dev_endpoint(f"/sleep?level={level}")

    async def wake_up(self) -> None:
        if not self.settings.enable_sleep_mode:
            return
        await self._post_dev_endpoint("/wake_up")
        await self.wait_awake()

    async def wait_awake(self, timeout_seconds: float | None = None) -> None:
        if not self.settings.enable_sleep_mode:
            return
        if timeout_seconds is None:
            timeout_seconds = _env_float_any(
                ("VLLM_JETSPEC_WAKE_TIMEOUT_SECONDS", "VLLM_DFLASH_WAKE_TIMEOUT_SECONDS"),
                120.0,
            )
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        poll_seconds = max(
            0.1,
            _env_float_any(
                (
                    "VLLM_JETSPEC_WAKE_POLL_SECONDS",
                    "VLLM_DFLASH_WAKE_POLL_SECONDS",
                ),
                0.5,
            ),
        )
        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.monotonic() < deadline:
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError(
                        f"vLLM exited with code {self.process.returncode}"
                    )
                try:
                    sleep_response = await client.get(
                        f"{self.settings.vllm_base_url}/is_sleeping"
                    )
                    sleeping = False
                    if sleep_response.status_code == 200:
                        payload = sleep_response.json()
                        sleeping = (
                            bool(
                                payload.get(
                                    "is_sleeping", payload.get("sleeping", False)
                                )
                            )
                            if isinstance(payload, dict)
                            else bool(payload)
                        )
                    if not sleeping:
                        ready_response = await client.get(
                            f"{self.settings.vllm_base_url}/v1/models"
                        )
                        if ready_response.status_code < 500:
                            return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(poll_seconds)
        raise TimeoutError("Timed out waiting for vLLM to wake up")

    async def load_lora_adapter(self, *, adapter_name: str, adapter_path: Path) -> bool:
        if not _live_lora_serving_enabled(self.settings):
            return False
        payload = {"lora_name": adapter_name, "lora_path": str(adapter_path)}
        async with httpx.AsyncClient(timeout=120.0) as client:
            await self._unload_lora_adapter(client, adapter_name)
            try:
                response = await client.post(
                    f"{self.settings.vllm_base_url}/v1/load_lora_adapter",
                    json=payload,
                )
                if response.status_code < 400:
                    return True
                logger.warning(
                    "vLLM rejected live LoRA adapter load status=%s body=%s",
                    response.status_code,
                    response.text[:500],
                )
            except Exception:
                logger.exception("Failed to load live LoRA adapter into vLLM")
        return False

    async def _unload_lora_adapter(
        self, client: httpx.AsyncClient, adapter_name: str
    ) -> None:
        try:
            await client.post(
                f"{self.settings.vllm_base_url}/v1/unload_lora_adapter",
                json={"lora_name": adapter_name},
            )
        except Exception:
            return

    async def is_sleeping(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.settings.vllm_base_url}/is_sleeping"
                )
                if response.status_code == 200:
                    payload = response.json()
                    if isinstance(payload, dict):
                        return bool(
                            payload.get("is_sleeping", payload.get("sleeping", False))
                        )
                    return bool(payload)
        except Exception:
            return False
        return False

    async def speculative_metrics(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.settings.vllm_base_url}/metrics")
                if response.status_code >= 400:
                    return {"available": False, "status_code": response.status_code}
                return _summarize_vllm_speculative_metrics(response.text)
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    async def _post_dev_endpoint(self, path: str) -> None:
        if not self.settings.enable_sleep_mode:
            return
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(f"{self.settings.vllm_base_url}{path}")
                if response.status_code >= 400:
                    response.raise_for_status()
        except httpx.HTTPStatusError:
            raise
        except Exception:
            return


class TrafficRecorder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self.persist_traffic = bool(getattr(settings, "persist_traffic", True))
        self.persist_suffix_cache = bool(
            getattr(settings, "persist_suffix_cache", True)
        )
        self.suffix_config = SuffixCacheConfig(
            max_tree_depth=_env_int("VLLM_SUFFIX_MAX_TREE_DEPTH", 24),
            max_cached_requests=_env_int("VLLM_SUFFIX_MAX_CACHED_REQUESTS", 10_000),
            max_spec_factor=_env_float("VLLM_SUFFIX_MAX_SPEC_FACTOR", 1.0),
            min_token_prob=_env_float("VLLM_SUFFIX_MIN_TOKEN_PROB", 0.1),
            min_match_tokens=_env_int("VLLM_SUFFIX_MIN_MATCH_TOKENS", 1),
        )
        self.suffix_record_max_tokens = _env_int_any(
            (
                "VLLM_JETSPEC_SUFFIX_RECORD_MAX_TOKENS",
                "VLLM_DFLASH_SUFFIX_RECORD_MAX_TOKENS",
            ),
            8192,
        )
        self.suffix_cache_save_every = _env_int_any(
            (
                "VLLM_JETSPEC_SUFFIX_CACHE_SAVE_EVERY",
                "VLLM_DFLASH_SUFFIX_CACHE_SAVE_EVERY",
            ),
            32,
        )
        self.suffix_cache_save_seconds = _env_float_any(
            (
                "VLLM_JETSPEC_SUFFIX_CACHE_SAVE_SECONDS",
                "VLLM_DFLASH_SUFFIX_CACHE_SAVE_SECONDS",
            ),
            30.0,
        )
        if not self.persist_traffic:
            _unlink_runtime_file(settings.traffic_path)
        if not self.persist_suffix_cache:
            _unlink_runtime_file(settings.suffix_cache_path)
        self._last_suffix_cache_save = time.monotonic()
        self._suffix_cache_records_since_save = 0
        self.tokenize_timeout_seconds = _env_float_any(
            (
                "VLLM_JETSPEC_SUFFIX_TOKENIZE_TIMEOUT_SECONDS",
                "VLLM_DFLASH_SUFFIX_TOKENIZE_TIMEOUT_SECONDS",
            ),
            30.0,
        )
        self.suffix_cache = (
            CpuSuffixCache.load(settings.suffix_cache_path, self.suffix_config)
            if self.persist_suffix_cache
            else CpuSuffixCache(self.suffix_config)
        )

    async def record(
        self, endpoint: str, request_payload: Any, response_payload: Any
    ) -> None:
        prompt = _extract_prompt(request_payload)
        completion = _extract_completion(response_payload)
        if not prompt and not completion:
            return
        row = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "endpoint": endpoint,
            "prompt": prompt,
            "completion": completion,
        }
        messages = _record_request_messages(request_payload)
        if messages:
            row["messages"] = messages
        token_ids, token_source, suffix_min_next_index = await self._suffix_token_ids(
            request_payload=request_payload,
            prompt=prompt,
            completion=completion,
        )
        if token_ids:
            row["suffix_token_count"] = len(token_ids)
            row["suffix_token_source"] = token_source
            row["suffix_token_ids"] = token_ids
            row["suffix_prompt_token_count"] = suffix_min_next_index
        with self._lock:
            self.settings.traffic_path.parent.mkdir(parents=True, exist_ok=True)
            with self.settings.traffic_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            if token_ids:
                self.suffix_cache.add_sequence(
                    token_ids,
                    request_id=_suffix_cache_request_id(row),
                    min_next_index=suffix_min_next_index,
                )
                self._suffix_cache_records_since_save += 1
                self._save_suffix_cache_if_due_locked()

    async def rebuild_suffix_cache(self, *, limit: int = 500) -> dict[str, Any]:
        rows = _tail_jsonl(self.settings.traffic_path, limit=max(1, limit))
        cache = CpuSuffixCache(self.suffix_config)
        sequences_added = 0
        tokens_added = 0
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.tokenize_timeout_seconds)
        ) as client:
            for row in rows:
                token_ids, min_next_index = await self._suffix_token_ids_from_stored_row(
                    row=row,
                    client=client,
                )
                if not token_ids:
                    continue
                cache.add_sequence(
                    token_ids,
                    request_id=_suffix_cache_request_id(row),
                    min_next_index=min_next_index,
                )
                sequences_added += 1
                tokens_added += len(token_ids)
        with self._lock:
            self.suffix_cache = cache
            self._save_suffix_cache_locked(force=True)
        return {
            "status": "rebuilt",
            "source": "vllm_tokenize_reconstructed_chat",
            "traffic_rows_scanned": len(rows),
            "sequences_added": sequences_added,
            "tokens_added": tokens_added,
            "suffix_cache_path": str(self.settings.suffix_cache_path),
            "suffix_cache_requests": self.suffix_cache.request_count,
            "suffix_cache_suffixes": self.suffix_cache.suffix_count,
            "boundary_aware": True,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "suffix_cache_path": str(self.settings.suffix_cache_path),
                "traffic_persistent": self.persist_traffic,
                "suffix_cache_persistent": self.persist_suffix_cache,
                "ephemeral_data_dir": str(
                    getattr(self.settings, "ephemeral_data_dir", "")
                ),
                "suffix_cache_backend": getattr(
                    self.suffix_cache, "backend_name", "python"
                ),
                "suffix_cache_requests": self.suffix_cache.request_count,
                "suffix_cache_suffixes": self.suffix_cache.suffix_count,
                "suffix_record_max_tokens": self.suffix_record_max_tokens,
                "suffix_cache_save_every": self.suffix_cache_save_every,
                "suffix_cache_save_seconds": self.suffix_cache_save_seconds,
                "suffix_cache_records_since_save": self._suffix_cache_records_since_save,
                "suffix_min_match_tokens": self.suffix_config.min_match_tokens,
                "suffix_min_token_prob": self.suffix_config.min_token_prob,
                "suffix_max_spec_factor": self.suffix_config.max_spec_factor,
                "suffix_max_tree_depth": self.suffix_config.max_tree_depth,
                "in_context_suffix_enabled": _env_bool(
                    "VLLM_JETSPEC_IN_CONTEXT_SUFFIX", True
                ),
                "in_context_suffix_bootstrap": _env_bool(
                    "VLLM_JETSPEC_IN_CONTEXT_SUFFIX_BOOTSTRAP", True
                ),
                "in_context_suffix_window": _env_int(
                    "VLLM_JETSPEC_IN_CONTEXT_SUFFIX_WINDOW", 8192
                ),
                "in_context_suffix_min_match_tokens": _env_int(
                    "VLLM_JETSPEC_IN_CONTEXT_SUFFIX_MIN_MATCH",
                    self.suffix_config.min_match_tokens,
                ),
            }

    async def _suffix_token_ids(
        self,
        *,
        request_payload: Any,
        prompt: str,
        completion: str,
    ) -> tuple[list[int], str, int]:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.tokenize_timeout_seconds)
        ) as client:
            prompt_tokens = await self._tokenize_request_payload(
                request_payload, client=client
            )
            completion_tokens = await self._completion_tokens_from_chat_delta(
                request_payload=request_payload,
                prompt_tokens=prompt_tokens,
                completion=completion,
                client=client,
            )
            token_source = "vllm_tokenize_chat_completion"
            if completion and not completion_tokens:
                completion_tokens = await self._tokenize_text(
                    completion, client=client
                )
                token_source = "vllm_tokenize"
            token_ids = prompt_tokens + completion_tokens
            if token_ids:
                trimmed, min_next_index = self._trim_suffix_record(
                    token_ids, len(prompt_tokens)
                )
                return trimmed, token_source, min_next_index
            fallback_prompt_tokens = (
                await self._tokenize_text(prompt, client=client) if prompt else []
            )
            fallback_tokens = fallback_prompt_tokens + completion_tokens
            if fallback_tokens:
                trimmed, min_next_index = self._trim_suffix_record(
                    fallback_tokens, len(fallback_prompt_tokens)
                )
                return (
                    trimmed,
                    "vllm_tokenize_text",
                    min_next_index,
                )
        return [], "", 1

    async def _suffix_token_ids_from_stored_row(
        self,
        *,
        row: dict[str, Any],
        client: httpx.AsyncClient,
    ) -> tuple[list[int], int]:
        stored_token_ids = row.get("suffix_token_ids")
        stored_prompt_token_count = _safe_int(row.get("suffix_prompt_token_count"))
        stored_token_source = str(row.get("suffix_token_source") or "")
        can_recompute_chat = bool(row.get("completion")) and (
            isinstance(row.get("messages"), list) or bool(row.get("prompt"))
        )
        if (
            isinstance(stored_token_ids, list)
            and stored_prompt_token_count is not None
            and (
                stored_token_source == "vllm_tokenize_chat_completion"
                or not can_recompute_chat
            )
        ):
            parsed_token_ids: list[int] = []
            for token_id in stored_token_ids:
                try:
                    parsed_token_ids.append(int(token_id))
                except (TypeError, ValueError):
                    continue
            if parsed_token_ids:
                return self._trim_suffix_record(
                    parsed_token_ids, stored_prompt_token_count
                )

        prompt = str(row.get("prompt") or "")
        completion = str(row.get("completion") or "")
        prompt_tokens = []
        prompt_payload: dict[str, Any] = {}
        if prompt:
            row_messages = row.get("messages")
            prompt_payload = {
                "model": self.settings.served_model_name,
                "messages": (
                    row_messages
                    if isinstance(row_messages, list)
                    else _messages_from_stored_prompt(prompt)
                ),
            }
            if _force_thinking_enabled():
                prompt_payload["chat_template_kwargs"] = {"enable_thinking": True}
            elif not _default_enable_thinking():
                prompt_payload["chat_template_kwargs"] = {"enable_thinking": False}
            prompt_tokens = await self._tokenize_request_payload(
                prompt_payload,
                client=client,
            )
        completion_tokens = await self._completion_tokens_from_chat_delta(
            request_payload=prompt_payload,
            prompt_tokens=prompt_tokens,
            completion=completion,
            client=client,
        )
        if completion and not completion_tokens:
            completion_tokens = await self._tokenize_text(completion, client=client)
        token_ids = prompt_tokens + completion_tokens
        if token_ids:
            return self._trim_suffix_record(token_ids, len(prompt_tokens))
        fallback_prompt_tokens = (
            await self._tokenize_text(prompt, client=client) if prompt else []
        )
        fallback_tokens = fallback_prompt_tokens + completion_tokens
        if fallback_tokens:
            return self._trim_suffix_record(fallback_tokens, len(fallback_prompt_tokens))
        return [], 1

    async def _tokenize_request_payload(
        self, request_payload: Any, *, client: httpx.AsyncClient
    ) -> list[int]:
        payload = self._minimal_tokenize_payload(request_payload)
        if not payload:
            return []
        return await self._tokenize(payload, client=client)

    async def _tokenize_text(
        self, text: str, *, client: httpx.AsyncClient
    ) -> list[int]:
        if not text:
            return []
        return await self._tokenize(
            {"model": self.settings.served_model_name, "prompt": text},
            client=client,
        )

    async def _completion_tokens_from_chat_delta(
        self,
        *,
        request_payload: Any,
        prompt_tokens: list[int],
        completion: str,
        client: httpx.AsyncClient,
    ) -> list[int]:
        if not completion or not prompt_tokens:
            return []
        if not _env_bool("VLLM_JETSPEC_SUFFIX_CHAT_DELTA", False):
            return []
        payload = self._minimal_tokenize_payload(request_payload)
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return []
        full_payload = dict(payload)
        full_payload["messages"] = list(messages) + [
            {"role": "assistant", "content": completion}
        ]
        full_payload.pop("add_generation_prompt", None)
        full_payload.pop("continue_final_message", None)
        full_tokens = await self._tokenize(full_payload, client=client)
        if not full_tokens:
            return []
        return self._completion_delta_tokens(prompt_tokens, full_tokens)

    @staticmethod
    def _completion_delta_tokens(
        prompt_tokens: list[int], full_tokens: list[int]
    ) -> list[int]:
        common_prefix = 0
        for prompt_token, full_token in zip(prompt_tokens, full_tokens):
            if prompt_token != full_token:
                break
            common_prefix += 1
        min_common_prefix = max(0, len(prompt_tokens) - 8)
        if common_prefix < min_common_prefix:
            return []
        return full_tokens[common_prefix:]

    async def _tokenize(
        self, payload: dict[str, Any], *, client: httpx.AsyncClient
    ) -> list[int]:
        try:
            response = await client.post(
                f"{self.settings.vllm_base_url}/tokenize", json=payload
            )
            if response.status_code >= 400:
                return []
            body = response.json()
        except Exception:
            return []
        tokens = body.get("tokens") if isinstance(body, dict) else None
        if not isinstance(tokens, list):
            return []
        parsed: list[int] = []
        for token in tokens:
            try:
                parsed.append(int(token))
            except (TypeError, ValueError):
                continue
        return parsed

    def _minimal_tokenize_payload(self, request_payload: Any) -> dict[str, Any]:
        if not isinstance(request_payload, dict):
            return {}
        model = str(request_payload.get("model") or self.settings.served_model_name)
        if isinstance(request_payload.get("messages"), list):
            payload: dict[str, Any] = {
                "model": model,
                "messages": request_payload["messages"],
            }
            for key in (
                "tools",
                "tool_choice",
                "documents",
                "chat_template_kwargs",
                "add_generation_prompt",
                "continue_final_message",
            ):
                if key in request_payload:
                    payload[key] = request_payload[key]
            return payload
        prompt = request_payload.get("prompt", request_payload.get("input", ""))
        if isinstance(prompt, list):
            prompt = "\n".join(str(item) for item in prompt)
        if prompt:
            return {"model": model, "prompt": str(prompt)}
        return {}

    def _trim_suffix_record_tokens(self, token_ids: list[int]) -> list[int]:
        if self.suffix_record_max_tokens <= 0:
            return token_ids
        return token_ids[-self.suffix_record_max_tokens :]

    def _trim_suffix_record(
        self, token_ids: list[int], prompt_token_count: int
    ) -> tuple[list[int], int]:
        original_length = len(token_ids)
        trimmed = self._trim_suffix_record_tokens(token_ids)
        removed = max(0, original_length - len(trimmed))
        min_next_index = max(1, int(prompt_token_count) - removed)
        return trimmed, min(min_next_index, len(trimmed))

    def _save_suffix_cache_if_due_locked(self) -> None:
        if self.suffix_cache_save_every <= 0 and self.suffix_cache_save_seconds <= 0:
            return
        now = time.monotonic()
        enough_records = (
            self.suffix_cache_save_every > 0
            and self._suffix_cache_records_since_save >= self.suffix_cache_save_every
        )
        enough_time = (
            self.suffix_cache_save_seconds > 0
            and now - self._last_suffix_cache_save >= self.suffix_cache_save_seconds
        )
        if enough_records or enough_time:
            self._save_suffix_cache_locked(force=True)

    def _save_suffix_cache_locked(self, *, force: bool = False) -> None:
        if not force and self._suffix_cache_records_since_save <= 0:
            return
        self.suffix_cache.save(self.settings.suffix_cache_path)
        self._suffix_cache_records_since_save = 0
        self._last_suffix_cache_save = time.monotonic()


class TrainingCoordinator:
    def __init__(self, settings: Settings, vllm: VllmProcess):
        self.settings = settings
        self.vllm = vllm
        self._lock = asyncio.Lock()
        self._inflight = 0
        self._last_activity = time.monotonic()
        self._trainer_process: asyncio.subprocess.Process | None = None
        self._last_train_run_id: str | None = None
        self._last_trainer_returncode: int | None = None
        self._last_train_started_at: float | None = None
        self._last_train_finished_at: float | None = None
        self._trainer_stop_requested = False
        self._trainer_failure_count = 0
        self._next_train_after_monotonic = 0.0
        self._last_hub_checkpoint_sync: dict[str, Any] = {}
        self._last_hub_checkpoint_sync_attempt = 0.0
        self._hub_checkpoint_sync_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        logger.info(
            "Training scheduler disabled; JetSpec and live LoRA training run only through explicit /jit/train routes"
        )

    async def stop(self) -> None:
        await self._stop_trainer()

    async def trainer_running(self) -> bool:
        async with self._lock:
            return (
                self._trainer_process is not None
                and self._trainer_process.returncode is None
            )

    async def enter_inference(self) -> None:
        self._schedule_hub_checkpoint_download(trigger="inference")
        async with self._lock:
            self._inflight += 1
            self._last_activity = time.monotonic()
            self._request_training_stop()
        await self._stop_trainer(
            terminate=_env_bool_any(
                (
                    "VLLM_JETSPEC_TRAIN_TERMINATE_ON_INFERENCE",
                    "VLLM_DFLASH_TRAIN_TERMINATE_ON_INFERENCE",
                ),
                True,
            ),
            graceful_timeout=_env_float_any(
                (
                    "VLLM_JETSPEC_TRAIN_STOP_WAIT_BEFORE_INFERENCE_SECONDS",
                    "VLLM_DFLASH_TRAIN_STOP_WAIT_BEFORE_INFERENCE_SECONDS",
                ),
                0.25,
            ),
        )
        await self.vllm.wake_up()

    async def exit_inference(self) -> None:
        async with self._lock:
            self._inflight = max(0, self._inflight - 1)
            self._last_activity = time.monotonic()

    async def status(self) -> dict[str, Any]:
        async with self._lock:
            trainer_running = (
                self._trainer_process is not None
                and self._trainer_process.returncode is None
            )
            idle_for_seconds = max(0.0, time.monotonic() - self._last_activity)
            traffic_snapshot = _traffic_snapshot(self.settings.traffic_path)
            training_data_path = Path(
                getattr(self.settings, "training_data_path", self.settings.traffic_path)
            )
            training_data_snapshot = _traffic_snapshot(training_data_path)
            live_training_path = Path(
                getattr(
                    self.settings,
                    "live_training_data_path",
                    self.settings.data_dir / "live_training.jsonl",
                )
            )
            live_training_snapshot = _traffic_snapshot(live_training_path)
            effective_training_snapshot = _effective_training_snapshot(
                training_data_path, self.settings.traffic_path
            )
            last_trained_traffic_snapshot = _read_last_trained_traffic_snapshot(
                self.settings.checkpoint_dir
            )
            live_lora_checkpoint_dir = Path(
                getattr(
                    self.settings,
                    "live_lora_checkpoint_dir",
                    self.settings.checkpoint_dir / "live_lora",
                )
            )
            last_trained_live_snapshot = _read_last_trained_traffic_snapshot(
                live_lora_checkpoint_dir
            )
            live_training_has_untrained_changes = (
                bool(getattr(self.settings, "live_training_enabled", False))
                and not _traffic_snapshot_matches(
                    live_training_snapshot,
                    last_trained_live_snapshot,
                )
                and live_training_snapshot.rows
                >= getattr(self.settings, "min_live_training_examples", 1)
            )
            training_has_untrained_changes = (
                not _traffic_snapshot_matches(
                    effective_training_snapshot,
                    last_trained_traffic_snapshot,
                )
                and effective_training_snapshot.rows
                >= self.settings.min_training_examples
            )
            trainer_parallel_mode = _trainer_parallel_mode(self.settings)
            trainer_world_size = _trainer_parallel_world_size(self.settings)
            return {
                "inflight": self._inflight,
                "idle_for_seconds": idle_for_seconds,
                "training_scheduler": "external",
                "training_trigger": "api",
                "next_training_in_seconds": None,
                "trainer_running": trainer_running,
                "trainer_pid": self._trainer_process.pid
                if trainer_running and self._trainer_process
                else None,
                "last_train_run_id": self._last_train_run_id,
                "last_trainer_returncode": self._last_trainer_returncode,
                "last_train_started_at": self._last_train_started_at,
                "last_train_finished_at": self._last_train_finished_at,
                "trainer_failure_count": self._trainer_failure_count,
                "trainer_failure_backoff_remaining_seconds": max(
                    0.0, self._next_train_after_monotonic - time.monotonic()
                ),
                "trainer_parallel_mode": trainer_parallel_mode,
                "trainer_world_size": trainer_world_size,
                "serving_model_parallel_size": _serving_model_parallel_size(
                    self.settings
                ),
                "api_training_enabled": True,
                "automatic_training_enabled": False,
                "legacy_training_enabled_env": self.settings.training_enabled,
                "stop_file": str(self.settings.stop_file),
                "persist_traffic": getattr(self.settings, "persist_traffic", True),
                "persist_suffix_cache": getattr(
                    self.settings, "persist_suffix_cache", True
                ),
                "ephemeral_data_dir": str(
                    getattr(self.settings, "ephemeral_data_dir", "")
                ),
                "traffic_path": str(self.settings.traffic_path),
                "traffic_examples": traffic_snapshot.rows,
                "traffic_snapshot": traffic_snapshot.as_dict(),
                "training_data_path": str(training_data_path),
                "training_examples": training_data_snapshot.rows,
                "training_data_snapshot": training_data_snapshot.as_dict(),
                "live_training_enabled": bool(
                    getattr(self.settings, "live_training_enabled", False)
                ),
                "live_training_data_path": str(live_training_path),
                "live_training_examples": live_training_snapshot.rows,
                "live_training_data_snapshot": live_training_snapshot.as_dict(),
                "last_trained_live_training_snapshot": last_trained_live_snapshot,
                "live_training_has_untrained_changes": live_training_has_untrained_changes,
                "min_live_training_examples": getattr(
                    self.settings, "min_live_training_examples", 1
                ),
                "live_lora_serving_enabled": _live_lora_serving_enabled(self.settings),
                "live_lora_checkpoint_dir": str(live_lora_checkpoint_dir),
                "latest_live_lora_checkpoint": _read_json_file(
                    live_lora_checkpoint_dir / "latest.json"
                ),
                "active_live_lora_adapter": _read_active_live_lora_adapter(
                    live_lora_checkpoint_dir
                ),
                "effective_training_snapshot": effective_training_snapshot.as_dict(),
                "last_trained_traffic_snapshot": last_trained_traffic_snapshot,
                "traffic_has_untrained_changes": training_has_untrained_changes,
                "training_has_untrained_changes": training_has_untrained_changes,
                "min_training_examples": self.settings.min_training_examples,
                "metrics_path": str(self.settings.metrics_path),
                "training_metrics": _summarize_training_metrics(
                    self.settings.metrics_path
                ),
                "checkpoint_dir": str(self.settings.checkpoint_dir),
                "latest_checkpoint": _read_json_file(
                    self.settings.checkpoint_dir / "latest.json"
                ),
                "hub_checkpoint_sync": hub_checkpoint_status_payload(
                    self.settings.model_name,
                    self.settings.checkpoint_dir,
                    served_model_name=self.settings.served_model_name,
                    last_sync=self._last_hub_checkpoint_sync,
                ),
                "speculative_config": self.settings.speculative_config,
                "spec_method": getattr(self.settings, "spec_method", ""),
                "jetspec_draft_model_resolution": (
                    self.settings.jetspec_draft_model_resolution.as_dict()
                    if getattr(self.settings, "jetspec_draft_model_resolution", None)
                    else None
                ),
                "proposer_stats_path": str(self.settings.proposer_stats_path),
                "proposer_stats": _read_json_file(self.settings.proposer_stats_path),
            }

    async def train_once(self, *, force: bool = False) -> int | None:
        return await self.train_jetspec_once(force=force)

    async def train_jetspec_once(self, *, force: bool = False) -> int | None:
        return await self._run_trainer_cycle(force=force, trigger="api")

    async def train_live_lora_once(
        self,
        *,
        force: bool = False,
        task_mix: str | None = None,
        max_steps: int | None = None,
        max_seq_len: int | None = None,
        loss_vocab_sample_size: int | None = None,
        lora_target_modules: str | None = None,
    ) -> int | None:
        handled, returncode = await self._run_live_lora_trainer_cycle(
            force=force,
            task_mix=task_mix,
            max_steps=max_steps,
            max_seq_len=max_seq_len,
            loss_vocab_sample_size=loss_vocab_sample_size,
            lora_target_modules=lora_target_modules,
        )
        return returncode if handled else None

    async def _run_trainer_cycle(
        self, *, force: bool = False, trigger: str
    ) -> int | None:
        if trigger != "api":
            raise RuntimeError(
                "JetSpec training is only allowed through the explicit /jit/train/jetspec API route"
            )
        await self._download_hub_checkpoint(trigger="before_train", force=force)
        training_data_snapshot: TrafficSnapshot | None = None
        async with self._lock:
            if self._inflight > 0:
                logger.info(
                    "Skipping JetSpec training cycle because inference is active"
                )
                return None
            if self._native_draft_method_needs_external_trainer():
                self._last_activity = time.monotonic()
                logger.info(
                    "Skipping native %s training cycle: set VLLM_JETSPEC_TRAINER_COMMAND "
                    "to continue a compatible draft head",
                    getattr(self.settings, "spec_method", "draft"),
                )
                return None
            training_data_path = Path(
                getattr(self.settings, "training_data_path", self.settings.traffic_path)
            )
            training_data_snapshot = _traffic_snapshot(training_data_path)
            effective_training_snapshot = _effective_training_snapshot(
                training_data_path, self.settings.traffic_path
            )
            if effective_training_snapshot.rows < self.settings.min_training_examples:
                self._last_activity = time.monotonic()
                logger.info(
                    "Skipping JetSpec training cycle before sleep: only %d effective training example(s), need at least %d path=%s",
                    effective_training_snapshot.rows,
                    self.settings.min_training_examples,
                    training_data_path,
                )
                return None
            if not force and _traffic_snapshot_matches(
                effective_training_snapshot,
                _read_last_trained_traffic_snapshot(self.settings.checkpoint_dir),
            ):
                self._last_activity = time.monotonic()
                logger.info(
                    "Skipping JetSpec training cycle before sleep: training data unchanged since last successful run "
                    "rows=%d size_bytes=%d mtime_ns=%d path=%s",
                    effective_training_snapshot.rows,
                    effective_training_snapshot.size_bytes,
                    effective_training_snapshot.mtime_ns,
                    training_data_path,
                )
                return None
            if (
                not force
                and time.monotonic() < self._next_train_after_monotonic
            ):
                remaining = self._next_train_after_monotonic - time.monotonic()
                self._last_activity = time.monotonic()
                logger.info(
                    "Skipping JetSpec training cycle during failure backoff remaining=%.1fs failures=%d",
                    remaining,
                    self._trainer_failure_count,
                )
                return None
            self._clear_training_stop()
            run_id = f"train-{uuid.uuid4().hex[:12]}"
            self._last_train_run_id = run_id
            self._last_trainer_returncode = None
            self._last_train_started_at = time.time()
            self._last_train_finished_at = None
            checkpoint_step_before = _latest_checkpoint_step(
                self.settings.checkpoint_dir
            )
        logger.info("Starting JetSpec training cycle run_id=%s", run_id)
        if getattr(self.settings, "enable_sleep_mode", True):
            await self.vllm.sleep(self.settings.sleep_level)
        else:
            logger.info(
                "Skipping vLLM sleep before JetSpec training run_id=%s because sleep mode is disabled",
                run_id,
            )
        aborted_after_sleep = False
        async with self._lock:
            if self._inflight > 0 or self.settings.stop_file.exists():
                logger.info(
                    "Aborting JetSpec training cycle run_id=%s after sleep; inference or stop requested",
                    run_id,
                )
                self._last_train_finished_at = time.time()
                aborted_after_sleep = True
        if aborted_after_sleep:
            if self.settings.wake_after_training:
                await self.vllm.wake_up()
            return None
        command = self._trainer_command(run_id)
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("OMP_NUM_THREADS", "1")
        logger.info(
            "Launching JetSpec trainer run_id=%s command=%s",
            run_id,
            shlex.join(command),
        )
        self._trainer_process = await asyncio.create_subprocess_exec(*command, env=env)
        returncode = await self._trainer_process.wait()
        checkpoint_step_after = _latest_checkpoint_step(self.settings.checkpoint_dir)
        checkpoint_advanced = (
            checkpoint_step_after is not None
            and checkpoint_step_after > (checkpoint_step_before or 0)
        )
        async with self._lock:
            stop_requested = self._trainer_stop_requested
            self._trainer_stop_requested = False
            self._last_trainer_returncode = returncode
            self._last_train_finished_at = time.time()
            if (
                (returncode == 0 or checkpoint_advanced)
                and training_data_snapshot is not None
                and (not self.settings.stop_file.exists() or checkpoint_advanced)
            ):
                _write_last_trained_traffic_snapshot(
                    self.settings.checkpoint_dir,
                    effective_training_snapshot,
                    run_id=run_id if returncode == 0 else f"{run_id}-partial",
                )
            if returncode == 0 or checkpoint_advanced:
                self._trainer_failure_count = 0
                self._next_train_after_monotonic = 0.0
            elif stop_requested:
                logger.info(
                    "JetSpec trainer stopped for inference/shutdown run_id=%s returncode=%s",
                    run_id,
                    returncode,
                )
                self._trainer_failure_count = 0
                self._next_train_after_monotonic = 0.0
            else:
                self._trainer_failure_count += 1
                delay = _trainer_failure_backoff_seconds(self._trainer_failure_count)
                self._next_train_after_monotonic = time.monotonic() + delay
                logger.warning(
                    "JetSpec trainer failed run_id=%s returncode=%s failures=%d next_retry_in=%.1fs",
                    run_id,
                    returncode,
                    self._trainer_failure_count,
                    delay,
                )
        logger.info(
            "JetSpec trainer exited run_id=%s returncode=%s", run_id, returncode
        )
        if returncode == 0 or checkpoint_advanced:
            await self._upload_hub_checkpoint(trigger="after_train")
        if self.settings.wake_after_training:
            await self.vllm.wake_up()
        return returncode

    def _hub_checkpoint_config(self) -> Any | None:
        model_name = getattr(self.settings, "model_name", "")
        if not model_name:
            return None
        return hub_checkpoint_config_from_env(
            model_name,
            served_model_name=getattr(self.settings, "served_model_name", None),
        )

    def _schedule_hub_checkpoint_download(self, *, trigger: str) -> None:
        config = self._hub_checkpoint_config()
        if (
            config is None
            or not config.enabled
            or not config.download_enabled
            or not config.sync_on_inference
        ):
            return
        now = time.monotonic()
        if now - self._last_hub_checkpoint_sync_attempt < config.sync_interval_seconds:
            return
        task = self._hub_checkpoint_sync_task
        if task is not None and not task.done():
            return
        self._hub_checkpoint_sync_task = asyncio.create_task(
            self._download_hub_checkpoint(trigger=trigger, force=True)
        )

    async def _download_hub_checkpoint(
        self, *, trigger: str, force: bool = False
    ) -> dict[str, Any]:
        config = self._hub_checkpoint_config()
        if config is None or not config.enabled or not config.download_enabled:
            result = {"status": "disabled", "direction": "download", "trigger": trigger}
            self._last_hub_checkpoint_sync = result
            return result
        now = time.monotonic()
        if (
            not force
            and trigger == "inference"
            and now - self._last_hub_checkpoint_sync_attempt
            < config.sync_interval_seconds
        ):
            return self._last_hub_checkpoint_sync
        self._last_hub_checkpoint_sync_attempt = now
        result = await asyncio.to_thread(
            hub_download_newer_checkpoint,
            model_name=self.settings.model_name,
            checkpoint_dir=self.settings.checkpoint_dir,
            config=config,
        )
        result["trigger"] = trigger
        result["checked_at"] = time.time()
        self._last_hub_checkpoint_sync = result
        if result.get("status") == "downloaded":
            logger.info(
                "Downloaded newer JetSpec checkpoint from Hugging Face repo=%s step=%s checkpoint=%s trigger=%s",
                config.repo_id,
                result.get("remote_step"),
                result.get("checkpoint_path"),
                trigger,
            )
        elif result.get("status") == "failed":
            logger.warning("JetSpec Hub checkpoint download failed: %s", result)
        return result

    async def _upload_hub_checkpoint(self, *, trigger: str) -> dict[str, Any]:
        config = self._hub_checkpoint_config()
        if config is None or not config.enabled or not config.upload_enabled:
            result = {"status": "disabled", "direction": "upload", "trigger": trigger}
            self._last_hub_checkpoint_sync = result
            return result
        result = await asyncio.to_thread(
            hub_upload_latest_checkpoint,
            model_name=self.settings.model_name,
            checkpoint_dir=self.settings.checkpoint_dir,
            config=config,
        )
        result["trigger"] = trigger
        result["checked_at"] = time.time()
        self._last_hub_checkpoint_sync = result
        if result.get("status") == "uploaded":
            logger.info(
                "Uploaded JetSpec checkpoint to Hugging Face repo=%s step=%s checkpoint=%s",
                config.repo_id,
                result.get("step"),
                result.get("checkpoint"),
            )
        elif result.get("status") == "failed":
            logger.warning("JetSpec Hub checkpoint upload failed: %s", result)
        return result

    async def _run_live_lora_trainer_cycle(
        self,
        *,
        force: bool = False,
        task_mix: str | None = None,
        max_steps: int | None = None,
        max_seq_len: int | None = None,
        loss_vocab_sample_size: int | None = None,
        lora_target_modules: str | None = None,
    ) -> tuple[bool, int | None]:
        if not bool(getattr(self.settings, "live_training_enabled", False)):
            return False, None
        live_training_path = Path(
            getattr(
                self.settings,
                "live_training_data_path",
                self.settings.data_dir / "live_training.jsonl",
            )
        )
        live_checkpoint_dir = Path(
            getattr(
                self.settings,
                "live_lora_checkpoint_dir",
                self.settings.checkpoint_dir / "live_lora",
            )
        )
        async with self._lock:
            if self._inflight > 0:
                return False, None
            live_snapshot = _traffic_snapshot(live_training_path)
            min_examples = int(getattr(self.settings, "min_live_training_examples", 1))
            if live_snapshot.rows < min_examples:
                return False, None
            if not force and _traffic_snapshot_matches(
                live_snapshot,
                _read_last_trained_traffic_snapshot(live_checkpoint_dir),
            ):
                return False, None
            if not force and time.monotonic() < self._next_train_after_monotonic:
                remaining = self._next_train_after_monotonic - time.monotonic()
                self._last_activity = time.monotonic()
                logger.info(
                    "Skipping live LoRA training cycle during failure backoff remaining=%.1fs failures=%d",
                    remaining,
                    self._trainer_failure_count,
                )
                return True, None
            self._clear_training_stop()
            run_id = f"live-train-{uuid.uuid4().hex[:12]}"
            self._last_train_run_id = run_id
            self._last_trainer_returncode = None
            self._last_train_started_at = time.time()
            self._last_train_finished_at = None
            checkpoint_step_before = _latest_checkpoint_step(live_checkpoint_dir)
        logger.info(
            "Starting live LoRA training cycle run_id=%s rows=%d path=%s",
            run_id,
            live_snapshot.rows,
            live_training_path,
        )
        if getattr(self.settings, "enable_sleep_mode", True):
            await self.vllm.sleep(self.settings.sleep_level)
        else:
            logger.info(
                "Skipping vLLM sleep before live LoRA training run_id=%s because sleep mode is disabled",
                run_id,
            )
        aborted_after_sleep = False
        async with self._lock:
            if self._inflight > 0 or self.settings.stop_file.exists():
                logger.info(
                    "Aborting live LoRA training cycle run_id=%s after sleep; inference or stop requested",
                    run_id,
                )
                self._last_train_finished_at = time.time()
                aborted_after_sleep = True
        if aborted_after_sleep:
            await self.vllm.wake_up()
            return True, None
        command = self._live_lora_trainer_command(
            run_id,
            task_mix=task_mix,
            max_steps=max_steps,
            max_seq_len=max_seq_len,
            loss_vocab_sample_size=loss_vocab_sample_size,
            lora_target_modules=lora_target_modules,
        )
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("OMP_NUM_THREADS", "1")
        logger.info(
            "Launching live LoRA trainer run_id=%s command=%s",
            run_id,
            shlex.join(command),
        )
        self._trainer_process = await asyncio.create_subprocess_exec(*command, env=env)
        returncode = await self._trainer_process.wait()
        checkpoint_step_after = _latest_checkpoint_step(live_checkpoint_dir)
        checkpoint_advanced = (
            checkpoint_step_after is not None
            and checkpoint_step_after > (checkpoint_step_before or 0)
        )
        async with self._lock:
            stop_requested = self._trainer_stop_requested
            self._trainer_stop_requested = False
            self._last_trainer_returncode = returncode
            self._last_train_finished_at = time.time()
            if (
                (returncode == 0 or checkpoint_advanced)
                and (not self.settings.stop_file.exists() or checkpoint_advanced)
            ):
                _write_last_trained_traffic_snapshot(
                    live_checkpoint_dir,
                    live_snapshot,
                    run_id=run_id if returncode == 0 else f"{run_id}-partial",
                )
            if returncode == 0 or checkpoint_advanced:
                self._trainer_failure_count = 0
                self._next_train_after_monotonic = 0.0
            elif stop_requested:
                logger.info(
                    "Live LoRA trainer stopped for inference/shutdown run_id=%s returncode=%s",
                    run_id,
                    returncode,
                )
                self._trainer_failure_count = 0
                self._next_train_after_monotonic = 0.0
            else:
                self._trainer_failure_count += 1
                delay = _trainer_failure_backoff_seconds(self._trainer_failure_count)
                self._next_train_after_monotonic = time.monotonic() + delay
                logger.warning(
                    "Live LoRA trainer failed run_id=%s returncode=%s failures=%d next_retry_in=%.1fs",
                    run_id,
                    returncode,
                    self._trainer_failure_count,
                    delay,
                )
        logger.info(
            "Live LoRA trainer exited run_id=%s returncode=%s", run_id, returncode
        )
        if returncode == 0 or checkpoint_advanced:
            await self._promote_live_lora_adapter()
        elif self.settings.wake_after_training or getattr(
            self.settings, "enable_sleep_mode", True
        ):
            await self.vllm.wake_up()
        return True, returncode

    async def _promote_live_lora_adapter(self) -> None:
        live_checkpoint_dir = Path(
            getattr(
                self.settings,
                "live_lora_checkpoint_dir",
                self.settings.checkpoint_dir / "live_lora",
            )
        )
        latest = _read_json_file(live_checkpoint_dir / "latest.json")
        adapter_path = latest.get("adapter_path")
        if not adapter_path:
            return
        adapter_dir = Path(str(adapter_path))
        if not adapter_dir.exists():
            logger.warning("Live LoRA adapter path is missing: %s", adapter_dir)
            return
        adapter_name = str(
            latest.get("adapter_name")
            or getattr(self.settings, "live_lora_adapter_name", "live-lora")
        )
        await self.vllm.wake_up()
        loaded = await self.vllm.load_lora_adapter(
            adapter_name=adapter_name,
            adapter_path=adapter_dir,
        )
        if loaded:
            _write_active_live_lora_adapter(
                live_checkpoint_dir,
                {
                    "adapter_name": adapter_name,
                    "adapter_path": str(adapter_dir),
                    "checkpoint_step": latest.get("step"),
                    "checkpoint_path": latest.get("checkpoint_path")
                    or latest.get("checkpoint"),
                    "loaded_at": time.time(),
                },
            )
        else:
            logger.warning(
                "Live LoRA adapter was trained but not loaded into vLLM adapter_name=%s adapter_path=%s",
                adapter_name,
                adapter_dir,
            )

    async def _stop_trainer(
        self,
        *,
        terminate: bool = True,
        graceful_timeout: float | None = None,
    ) -> None:
        process = self._trainer_process
        if process is None or process.returncode is not None:
            return
        logger.info("Requesting JetSpec trainer stop pid=%s", process.pid)
        self._request_training_stop()
        async with self._lock:
            if self._trainer_process is process and process.returncode is None:
                self._trainer_stop_requested = True
        if graceful_timeout is None:
            graceful_timeout = _env_float_any(
                (
                    "VLLM_JETSPEC_TRAIN_STOP_GRACE_SECONDS",
                    "VLLM_DFLASH_TRAIN_STOP_GRACE_SECONDS",
                ),
                30.0,
            )
        graceful_timeout = max(0.0, graceful_timeout)
        if graceful_timeout > 0:
            try:
                await asyncio.wait_for(process.wait(), timeout=graceful_timeout)
                return
            except asyncio.TimeoutError:
                pass
        if not terminate:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    def _request_training_stop(self) -> None:
        self.settings.stop_file.parent.mkdir(parents=True, exist_ok=True)
        self.settings.stop_file.write_text(str(time.time()), encoding="utf-8")

    def _clear_training_stop(self) -> None:
        try:
            self.settings.stop_file.unlink()
        except FileNotFoundError:
            pass

    def _native_draft_method_needs_external_trainer(self) -> bool:
        speculative_config = getattr(self.settings, "speculative_config", {}) or {}
        spec_method = str(getattr(self.settings, "spec_method", "")).strip().lower()
        return (
            speculative_config.get("method") == "dflash"
            and spec_method in JETSPEC_SPEC_METHODS
            and not str(getattr(self.settings, "trainer_command", "")).strip()
        )

    def _trainer_command(self, run_id: str) -> list[str]:
        configured_command = str(getattr(self.settings, "trainer_command", "")).strip()
        if configured_command:
            values = _trainer_command_values(self.settings, run_id)
            for key, value in values.items():
                configured_command = configured_command.replace("{" + key + "}", value)
            return shlex.split(configured_command)
        trainer_args = [
            "--model-name",
            self.settings.model_name,
            "--data-path",
            str(
                getattr(
                    self.settings,
                    "training_data_path",
                    self.settings.traffic_path,
                )
            ),
            "--checkpoint-dir",
            str(self.settings.checkpoint_dir),
            "--suffix-cache-path",
            str(self.settings.suffix_cache_path),
            "--token-source-mode",
            _env_first_str(
                "VLLM_JETSPEC_TRAIN_TOKEN_SOURCE_MODE",
                "VLLM_DFLASH_TRAIN_TOKEN_SOURCE_MODE",
                default="text",
            ),
            *_bool_cli_option(
                "--staged-finetune",
                _env_bool_any(
                    ("VLLM_JETSPEC_STAGED_FINETUNE", "VLLM_DFLASH_STAGED_FINETUNE"),
                    False,
                ),
            ),
            "--device",
            _env_first_str(
                "VLLM_JETSPEC_TRAIN_DEVICE",
                "VLLM_DFLASH_TRAIN_DEVICE",
                default="cuda:0",
            ),
            "--parallel-mode",
            _trainer_parallel_mode(self.settings),
            "--max-steps",
            str(self.settings.train_steps_per_idle),
            "--pretrain-max-steps",
            _env_first_str(
                "VLLM_JETSPEC_PRETRAIN_MAX_STEPS",
                "VLLM_DFLASH_PRETRAIN_MAX_STEPS",
                default="0",
            ),
            "--finetune-max-steps",
            _env_first_str(
                "VLLM_JETSPEC_FINETUNE_MAX_STEPS",
                "VLLM_DFLASH_FINETUNE_MAX_STEPS",
                default="0",
            ),
            "--stop-file",
            str(self.settings.stop_file),
            "--metrics-path",
            str(self.settings.metrics_path),
            "--num-speculative-tokens",
            str(
                _env_int_any(
                    (
                        "VLLM_JETSPEC_NUM_SPECULATIVE_TOKENS",
                        "VLLM_DFLASH_NUM_SPECULATIVE_TOKENS",
                    ),
                    8,
                )
            ),
            "--learning-rate",
            _env_first_str("VLLM_JETSPEC_LR", "VLLM_DFLASH_LR", default="0.01"),
            "--train-epochs",
            _env_first_str(
                "VLLM_JETSPEC_TRAIN_EPOCHS",
                "VLLM_DFLASH_TRAIN_EPOCHS",
                default="1",
            ),
            "--pretrain-epochs",
            _env_first_str(
                "VLLM_JETSPEC_PRETRAIN_EPOCHS",
                "VLLM_DFLASH_PRETRAIN_EPOCHS",
                "VLLM_JETSPEC_TRAIN_EPOCHS",
                "VLLM_DFLASH_TRAIN_EPOCHS",
                default="1",
            ),
            "--finetune-epochs",
            _env_first_str(
                "VLLM_JETSPEC_FINETUNE_EPOCHS",
                "VLLM_DFLASH_FINETUNE_EPOCHS",
                "VLLM_JETSPEC_TRAIN_EPOCHS",
                "VLLM_DFLASH_TRAIN_EPOCHS",
                default="1",
            ),
            "--pretrain-live-step-ratio",
            _env_first_str(
                "VLLM_JETSPEC_PRETRAIN_LIVE_STEP_RATIO",
                "VLLM_DFLASH_PRETRAIN_LIVE_STEP_RATIO",
                default="0",
            ),
            "--batch-size",
            _env_first_str("VLLM_JETSPEC_BATCH_SIZE", "VLLM_DFLASH_BATCH_SIZE", default="4"),
            "--max-seq-len",
            _env_first_str(
                "VLLM_JETSPEC_MAX_SEQ_LEN", "VLLM_DFLASH_MAX_SEQ_LEN", default="256"
            ),
            "--hidden-size",
            _env_first_str(
                "VLLM_JETSPEC_HIDDEN_SIZE", "VLLM_DFLASH_HIDDEN_SIZE", default="32"
            ),
            "--adapter-architecture",
            _env_first_str(
                "VLLM_JETSPEC_ADAPTER_ARCHITECTURE",
                "VLLM_DFLASH_ADAPTER_ARCHITECTURE",
                default="pooled_gru",
            ),
            "--max-examples-per-traffic",
            _env_first_str(
                "VLLM_JETSPEC_MAX_EXAMPLES_PER_TRAFFIC",
                "VLLM_DFLASH_MAX_EXAMPLES_PER_TRAFFIC",
                default="4",
            ),
            "--example-stride",
            _env_first_str(
                "VLLM_JETSPEC_EXAMPLE_STRIDE",
                "VLLM_DFLASH_EXAMPLE_STRIDE",
                default="1",
            ),
            "--min-examples",
            _env_first_str(
                "VLLM_JETSPEC_MIN_EXAMPLES", "VLLM_DFLASH_MIN_EXAMPLES", default="8"
            ),
            *_bool_cli_option(
                "--auto-full-pass-each-run",
                _env_bool_any(
                    (
                        "VLLM_JETSPEC_AUTO_FULL_PASS_EACH_RUN",
                        "VLLM_DFLASH_AUTO_FULL_PASS_EACH_RUN",
                    ),
                    True,
                ),
            ),
            *_bool_cli_option(
                "--reset-on-data-source-change",
                _env_bool_any(
                    (
                        "VLLM_JETSPEC_RESET_ON_DATA_SOURCE_CHANGE",
                        "VLLM_DFLASH_RESET_ON_DATA_SOURCE_CHANGE",
                    ),
                    False,
                ),
            ),
            "--checkpoint-every",
            _env_first_str(
                "VLLM_JETSPEC_CHECKPOINT_EVERY",
                "VLLM_DFLASH_CHECKPOINT_EVERY",
                default="16",
            ),
            "--keep-last-checkpoints",
            _env_first_str(
                "VLLM_JETSPEC_KEEP_LAST_CHECKPOINTS",
                "VLLM_DFLASH_KEEP_LAST_CHECKPOINTS",
                default="2",
            ),
            *_bool_cli_option(
                "--quality-gate",
                _env_bool_any(
                    ("VLLM_JETSPEC_QUALITY_GATE", "VLLM_DFLASH_QUALITY_GATE"),
                    True,
                ),
            ),
            "--quality-eval-examples",
            _env_first_str(
                "VLLM_JETSPEC_QUALITY_EVAL_EXAMPLES",
                "VLLM_DFLASH_QUALITY_EVAL_EXAMPLES",
                default="128",
            ),
            "--quality-min-delta",
            _env_first_str(
                "VLLM_JETSPEC_QUALITY_MIN_DELTA",
                "VLLM_DFLASH_QUALITY_MIN_DELTA",
                default="0",
            ),
            *_bool_cli_option(
                "--promote-intermediate-checkpoints",
                _env_bool_any(
                    (
                        "VLLM_JETSPEC_PROMOTE_INTERMEDIATE_CHECKPOINTS",
                        "VLLM_DFLASH_PROMOTE_INTERMEDIATE_CHECKPOINTS",
                    ),
                    False,
                ),
            ),
            "--run-id",
            run_id,
            "--log-every",
            str(self.settings.train_log_every),
        ]
        training_data_path = getattr(
            self.settings,
            "training_data_path",
            self.settings.traffic_path,
        )
        if (
            _mix_live_traffic_enabled()
            and Path(training_data_path) != self.settings.traffic_path
        ):
            trainer_args.extend(["--extra-data-path", str(self.settings.traffic_path)])
        world_size = _trainer_parallel_world_size(self.settings)
        trainer_module = "vllm_dflash_jit.trainer"
        if world_size <= 1 or _trainer_parallel_mode(self.settings) == "none":
            return [sys.executable, "-m", trainer_module, *trainer_args]
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            str(world_size),
            "-m",
            trainer_module,
            *trainer_args,
        ]

    def _live_lora_trainer_command(
        self,
        run_id: str,
        *,
        task_mix: str | None = None,
        max_steps: int | None = None,
        max_seq_len: int | None = None,
        loss_vocab_sample_size: int | None = None,
        lora_target_modules: str | None = None,
    ) -> list[str]:
        live_checkpoint_dir = Path(
            getattr(
                self.settings,
                "live_lora_checkpoint_dir",
                self.settings.checkpoint_dir / "live_lora",
            )
        )
        trainer_args = [
            "--model-name",
            str(self.settings.model_name),
            "--data-path",
            str(
                getattr(
                    self.settings,
                    "live_training_data_path",
                    self.settings.data_dir / "live_training.jsonl",
                )
            ),
            "--checkpoint-dir",
            str(live_checkpoint_dir),
            "--adapter-name",
            str(getattr(self.settings, "live_lora_adapter_name", "live-lora")),
            "--task-mix",
            task_mix
            if task_mix is not None
            else _env_first_str(
                "VLLM_JETSPEC_LIVE_TRAIN_TASK_MIX",
                "VLLM_DFLASH_LIVE_TRAIN_TASK_MIX",
                default="mixed",
            ),
            "--device",
            _env_first_str(
                "VLLM_JETSPEC_LIVE_TRAIN_DEVICE",
                "VLLM_DFLASH_LIVE_TRAIN_DEVICE",
                "VLLM_JETSPEC_TRAIN_DEVICE",
                "VLLM_DFLASH_TRAIN_DEVICE",
                default="cuda",
            ),
            "--parallel-mode",
            _live_trainer_parallel_mode(self.settings),
            "--max-steps",
            str(max(0, int(max_steps)))
            if max_steps is not None
            else _env_first_str(
                "VLLM_JETSPEC_LIVE_TRAIN_STEPS",
                "VLLM_DFLASH_LIVE_TRAIN_STEPS",
                default="0",
            ),
            "--train-epochs",
            _env_first_str(
                "VLLM_JETSPEC_LIVE_TRAIN_EPOCHS",
                "VLLM_DFLASH_LIVE_TRAIN_EPOCHS",
                default="1",
            ),
            "--batch-size",
            _env_first_str(
                "VLLM_JETSPEC_LIVE_BATCH_SIZE",
                "VLLM_DFLASH_LIVE_BATCH_SIZE",
                default="1",
            ),
            "--gradient-accumulation-steps",
            _env_first_str(
                "VLLM_JETSPEC_LIVE_GRADIENT_ACCUMULATION_STEPS",
                "VLLM_DFLASH_LIVE_GRADIENT_ACCUMULATION_STEPS",
                default="1",
            ),
            "--max-seq-len",
            str(max(1, int(max_seq_len)))
            if max_seq_len is not None
            else _env_first_str(
                "VLLM_JETSPEC_LIVE_MAX_SEQ_LEN",
                "VLLM_DFLASH_LIVE_MAX_SEQ_LEN",
                default="2048",
            ),
            "--learning-rate",
            _env_first_str(
                "VLLM_JETSPEC_LIVE_LR",
                "VLLM_DFLASH_LIVE_LR",
                default="0.0002",
            ),
            "--weight-decay",
            _env_first_str(
                "VLLM_JETSPEC_LIVE_WEIGHT_DECAY",
                "VLLM_DFLASH_LIVE_WEIGHT_DECAY",
                default="0",
            ),
            "--max-grad-norm",
            _env_first_str(
                "VLLM_JETSPEC_LIVE_MAX_GRAD_NORM",
                "VLLM_DFLASH_LIVE_MAX_GRAD_NORM",
                default="1",
            ),
            "--dpo-beta",
            _env_first_str(
                "VLLM_JETSPEC_LIVE_DPO_BETA",
                "VLLM_DFLASH_LIVE_DPO_BETA",
                default="0.1",
            ),
            "--loss-vocab-sample-size",
            str(max(0, int(loss_vocab_sample_size)))
            if loss_vocab_sample_size is not None
            else _env_first_str(
                "VLLM_JETSPEC_LIVE_LOSS_VOCAB_SAMPLE_SIZE",
                "VLLM_DFLASH_LIVE_LOSS_VOCAB_SAMPLE_SIZE",
                default="0",
            ),
            "--lora-r",
            _env_first_str(
                "VLLM_JETSPEC_LIVE_LORA_R",
                "VLLM_DFLASH_LIVE_LORA_R",
                default="16",
            ),
            "--lora-alpha",
            _env_first_str(
                "VLLM_JETSPEC_LIVE_LORA_ALPHA",
                "VLLM_DFLASH_LIVE_LORA_ALPHA",
                default="32",
            ),
            "--lora-dropout",
            _env_first_str(
                "VLLM_JETSPEC_LIVE_LORA_DROPOUT",
                "VLLM_DFLASH_LIVE_LORA_DROPOUT",
                default="0.05",
            ),
            "--lora-target-modules",
            lora_target_modules
            if lora_target_modules is not None
            else _env_first_str(
                "VLLM_JETSPEC_LIVE_LORA_TARGET_MODULES",
                "VLLM_DFLASH_LIVE_LORA_TARGET_MODULES",
                default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,qkv_proj,gate_up_proj,wq,wk,wv,wo",
            ),
            *_bool_cli_option(
                "--include-expert-lora",
                _env_bool_any(
                    (
                        "VLLM_JETSPEC_LIVE_INCLUDE_EXPERT_LORA",
                        "VLLM_DFLASH_LIVE_INCLUDE_EXPERT_LORA",
                    ),
                    False,
                ),
            ),
            "--quantization",
            _env_first_str(
                "VLLM_JETSPEC_LIVE_TRAIN_QUANTIZATION",
                "VLLM_DFLASH_LIVE_TRAIN_QUANTIZATION",
                default="auto",
            ),
            "--torch-dtype",
            _env_first_str(
                "VLLM_JETSPEC_LIVE_TORCH_DTYPE",
                "VLLM_DFLASH_LIVE_TORCH_DTYPE",
                default="bfloat16",
            ),
            *_bool_cli_option("--trust-remote-code", self.settings.trust_remote_code),
            *_bool_cli_option(
                "--save-optimizer-state",
                _env_bool_any(
                    (
                        "VLLM_JETSPEC_LIVE_SAVE_OPTIMIZER_STATE",
                        "VLLM_DFLASH_LIVE_SAVE_OPTIMIZER_STATE",
                    ),
                    True,
                ),
            ),
            "--checkpoint-every",
            _env_first_str(
                "VLLM_JETSPEC_LIVE_CHECKPOINT_EVERY",
                "VLLM_DFLASH_LIVE_CHECKPOINT_EVERY",
                default="16",
            ),
            "--keep-last-checkpoints",
            _env_first_str(
                "VLLM_JETSPEC_LIVE_KEEP_LAST_CHECKPOINTS",
                "VLLM_DFLASH_LIVE_KEEP_LAST_CHECKPOINTS",
                default="2",
            ),
            "--stop-file",
            str(self.settings.stop_file),
            "--metrics-path",
            str(self.settings.metrics_path),
            "--run-id",
            run_id,
            "--log-every",
            _env_first_str(
                "VLLM_JETSPEC_LIVE_TRAIN_LOG_EVERY",
                "VLLM_DFLASH_LIVE_TRAIN_LOG_EVERY",
                "VLLM_JETSPEC_TRAIN_LOG_EVERY",
                "VLLM_DFLASH_TRAIN_LOG_EVERY",
                default=str(self.settings.train_log_every),
            ),
        ]
        return [sys.executable, "-m", "vllm_dflash_jit.live_lora_trainer", *trainer_args]


def _trainer_command_values(settings: Settings, run_id: str) -> dict[str, str]:
    draft_resolution = getattr(settings, "jetspec_draft_model_resolution", None)
    default_subdir = "jetspec"
    checked_paths = (
        list(draft_resolution.checked_local_paths)
        if draft_resolution is not None
        else []
    )
    draft_model_dir = (
        draft_resolution.local_path
        if draft_resolution is not None and draft_resolution.local_path is not None
        else (
            checked_paths[0]
            if checked_paths
            else settings.checkpoint_dir / default_subdir
        )
    )
    draft_model = draft_resolution.model if draft_resolution is not None else ""
    suffix_cache_path = getattr(
        settings,
        "suffix_cache_path",
        Path(settings.traffic_path).with_name("suffix_cache.json"),
    )
    return {
        "model_name": settings.model_name,
        "traffic_path": str(settings.traffic_path),
        "data_path": str(
            getattr(settings, "training_data_path", settings.traffic_path)
        ),
        "training_data_path": str(
            getattr(settings, "training_data_path", settings.traffic_path)
        ),
        "checkpoint_dir": str(settings.checkpoint_dir),
        "suffix_cache_path": str(suffix_cache_path),
        "draft_model_dir": str(draft_model_dir),
        "draft_model": draft_model,
        "jetspec_draft_model": draft_model,
        "jetspec_draft_head": draft_model,
        "jetspec_draft_model_dir": str(draft_model_dir),
        "jetspec_draft_head_dir": str(draft_model_dir),
        "metrics_path": str(settings.metrics_path),
        "stop_file": str(settings.stop_file),
        "run_id": run_id,
        "train_steps": str(settings.train_steps_per_idle),
        "train_epochs": _env_first_str(
            "VLLM_JETSPEC_TRAIN_EPOCHS", "VLLM_DFLASH_TRAIN_EPOCHS", default="1"
        ),
        "pretrain_epochs": _env_first_str(
            "VLLM_JETSPEC_PRETRAIN_EPOCHS",
            "VLLM_DFLASH_PRETRAIN_EPOCHS",
            "VLLM_JETSPEC_TRAIN_EPOCHS",
            "VLLM_DFLASH_TRAIN_EPOCHS",
            default="1",
        ),
        "finetune_epochs": _env_first_str(
            "VLLM_JETSPEC_FINETUNE_EPOCHS",
            "VLLM_DFLASH_FINETUNE_EPOCHS",
            "VLLM_JETSPEC_TRAIN_EPOCHS",
            "VLLM_DFLASH_TRAIN_EPOCHS",
            default="1",
        ),
        "pretrain_live_step_ratio": _env_first_str(
            "VLLM_JETSPEC_PRETRAIN_LIVE_STEP_RATIO",
            "VLLM_DFLASH_PRETRAIN_LIVE_STEP_RATIO",
            default="0",
        ),
        "pretrain_max_steps": _env_first_str(
            "VLLM_JETSPEC_PRETRAIN_MAX_STEPS",
            "VLLM_DFLASH_PRETRAIN_MAX_STEPS",
            default="0",
        ),
        "finetune_max_steps": _env_first_str(
            "VLLM_JETSPEC_FINETUNE_MAX_STEPS",
            "VLLM_DFLASH_FINETUNE_MAX_STEPS",
            default="0",
        ),
        "staged_finetune": str(
            _env_bool_any(
                ("VLLM_JETSPEC_STAGED_FINETUNE", "VLLM_DFLASH_STAGED_FINETUNE"),
                False,
            )
        ).lower(),
        "learning_rate": _env_first_str(
            "VLLM_JETSPEC_LR", "VLLM_DFLASH_LR", default="0.01"
        ),
        "batch_size": _env_first_str(
            "VLLM_JETSPEC_BATCH_SIZE", "VLLM_DFLASH_BATCH_SIZE", default="4"
        ),
        "max_seq_len": _env_first_str(
            "VLLM_JETSPEC_MAX_SEQ_LEN", "VLLM_DFLASH_MAX_SEQ_LEN", default="256"
        ),
        "hidden_size": _env_first_str(
            "VLLM_JETSPEC_HIDDEN_SIZE", "VLLM_DFLASH_HIDDEN_SIZE", default="32"
        ),
        "adapter_architecture": _env_first_str(
            "VLLM_JETSPEC_ADAPTER_ARCHITECTURE",
            "VLLM_DFLASH_ADAPTER_ARCHITECTURE",
            default="pooled_gru",
        ),
        "max_examples_per_traffic": _env_first_str(
            "VLLM_JETSPEC_MAX_EXAMPLES_PER_TRAFFIC",
            "VLLM_DFLASH_MAX_EXAMPLES_PER_TRAFFIC",
            default="4",
        ),
        "example_stride": _env_first_str(
            "VLLM_JETSPEC_EXAMPLE_STRIDE",
            "VLLM_DFLASH_EXAMPLE_STRIDE",
            default="1",
        ),
        "min_examples": _env_first_str(
            "VLLM_JETSPEC_MIN_EXAMPLES", "VLLM_DFLASH_MIN_EXAMPLES", default="8"
        ),
        "checkpoint_every": _env_first_str(
            "VLLM_JETSPEC_CHECKPOINT_EVERY",
            "VLLM_DFLASH_CHECKPOINT_EVERY",
            default="16",
        ),
        "keep_last_checkpoints": _env_first_str(
            "VLLM_JETSPEC_KEEP_LAST_CHECKPOINTS",
            "VLLM_DFLASH_KEEP_LAST_CHECKPOINTS",
            default="2",
        ),
        "quality_eval_examples": _env_first_str(
            "VLLM_JETSPEC_QUALITY_EVAL_EXAMPLES",
            "VLLM_DFLASH_QUALITY_EVAL_EXAMPLES",
            default="128",
        ),
        "quality_min_delta": _env_first_str(
            "VLLM_JETSPEC_QUALITY_MIN_DELTA",
            "VLLM_DFLASH_QUALITY_MIN_DELTA",
            default="0",
        ),
        "promote_intermediate_checkpoints": str(
            _env_bool_any(
                (
                    "VLLM_JETSPEC_PROMOTE_INTERMEDIATE_CHECKPOINTS",
                    "VLLM_DFLASH_PROMOTE_INTERMEDIATE_CHECKPOINTS",
                ),
                False,
            )
        ).lower(),
        "num_speculative_tokens": str(
            _env_int_any(
                (
                    "VLLM_JETSPEC_NUM_SPECULATIVE_TOKENS",
                    "VLLM_DFLASH_NUM_SPECULATIVE_TOKENS",
                ),
                8,
            )
        ),
        "train_parallel_mode": _trainer_parallel_mode(settings),
        "train_world_size": str(_trainer_parallel_world_size(settings)),
        "serving_model_parallel_size": str(_serving_model_parallel_size(settings)),
        "tensor_parallel_size": str(
            _vllm_parallel_sizes(getattr(settings, "vllm_extra_args", ""))[
                "tensor_parallel_size"
            ]
        ),
        "pipeline_parallel_size": str(
            _vllm_parallel_sizes(getattr(settings, "vllm_extra_args", ""))[
                "pipeline_parallel_size"
            ]
        ),
    }


def _extract_prompt(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    if "messages" in payload:
        return "\n".join(
            _message_text(message) for message in payload.get("messages") or []
        )
    prompt = payload.get("prompt") or payload.get("input") or ""
    if isinstance(prompt, list):
        return "\n".join(str(item) for item in prompt)
    return str(prompt)


def _message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return str(message)
    content = message.get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        content_text = "".join(parts)
    else:
        content_text = str(content)
    role = message.get("role", "")
    return f"{role}: {content_text}" if role else content_text


def _messages_from_stored_prompt(prompt: str) -> list[dict[str, str]]:
    prompt = str(prompt or "")
    if not prompt:
        return []

    known_roles = {
        "assistant",
        "developer",
        "system",
        "tool",
        "user",
    }
    messages: list[dict[str, str]] = []
    current_role: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_role, current_lines
        if current_role is None:
            return
        messages.append(
            {"role": current_role, "content": "\n".join(current_lines).strip()}
        )
        current_role = None
        current_lines = []

    for line in prompt.splitlines():
        prefix, separator, rest = line.partition(":")
        role = prefix.strip().lower()
        if separator and role in known_roles:
            flush()
            current_role = role
            current_lines = [rest.lstrip()]
            continue
        if current_role is None:
            current_role = "user"
            current_lines = [line]
        else:
            current_lines.append(line)
    flush()

    if not messages:
        return [{"role": "user", "content": prompt}]
    return messages


def _record_request_messages(payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        return []
    messages: list[dict[str, str]] = []
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = _content_text(message.get("content", ""))
        if not content:
            continue
        messages.append(
            {"role": str(message.get("role") or "user"), "content": content}
        )
    return messages


def _suffix_cache_request_id(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        key = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        key = str(row.get("prompt") or "")
    if not key:
        return str(row.get("id") or uuid.uuid4())
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"prompt-{digest}"


def _extract_completion(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                parts: list[str] = []
                for key in ("reasoning", "reasoning_content", "content"):
                    value = message.get(key)
                    if value:
                        parts.append(_content_text(value))
                return "\n".join(part for part in parts if part)
            return str(choice.get("text") or "")
    output_text = payload.get("output_text")
    if output_text:
        return str(output_text)
    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if isinstance(item, dict):
                for content in item.get("content") or []:
                    if isinstance(content, dict):
                        parts.append(str(content.get("text") or ""))
        return "".join(parts)
    return ""


class StreamingCompletionCapture:
    def __init__(self, max_bytes: int | None = None):
        self.max_bytes = max_bytes or _env_int_any(
            (
                "VLLM_JETSPEC_MAX_STREAM_RECORD_BYTES",
                "VLLM_DFLASH_MAX_STREAM_RECORD_BYTES",
            ),
            4 * 1024 * 1024,
        )
        self._buffer = ""
        self._seen_bytes = 0
        self._parts: list[str] = []

    @property
    def completion(self) -> str:
        return "".join(self._parts)

    def feed(self, chunk: bytes) -> None:
        if self._seen_bytes >= self.max_bytes:
            return
        remaining = self.max_bytes - self._seen_bytes
        chunk = chunk[:remaining]
        self._seen_bytes += len(chunk)
        self._buffer += chunk.decode("utf-8", errors="ignore")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume_sse_line(line.strip())

    def finish(self) -> None:
        if self._buffer.strip():
            self._consume_sse_line(self._buffer.strip())
        self._buffer = ""

    def _consume_sse_line(self, line: str) -> None:
        if not line.startswith("data:"):
            return
        data = line[5:].strip()
        if not data or data == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return
        self._consume_stream_payload(payload)

    def _consume_stream_payload(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        choices = payload.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    self._append_content(delta.get("reasoning"))
                    self._append_content(delta.get("reasoning_content"))
                    self._append_content(delta.get("content"))
                self._append_content(choice.get("text"))

        if payload.get("type") in {
            "response.output_text.delta",
            "response.reasoning_text.delta",
            "response.reasoning.delta",
        }:
            self._append_content(payload.get("delta"))
        self._append_content(payload.get("output_text"))
        self._append_content(payload.get("reasoning"))
        self._append_content(payload.get("reasoning_content"))

    def _append_content(self, content: Any) -> None:
        if isinstance(content, str):
            self._parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    self._append_content(item.get("text") or item.get("content"))
                else:
                    self._append_content(item)


def _completion_payload(completion: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": completion}}]}


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


def _filtered_request_headers(request: Request) -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def _filtered_response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def _request_enables_thinking(request_payload: Any) -> bool:
    if not isinstance(request_payload, dict):
        return False
    chat_template_kwargs = request_payload.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        return False
    return bool(chat_template_kwargs.get("enable_thinking"))


def _move_reasoning_to_content_for_no_thinking(message: dict[str, Any]) -> bool:
    reasoning_parts: list[str] = []
    for key in ("reasoning", "reasoning_content"):
        value = message.pop(key, None)
        if value is None:
            continue
        text = _content_text(value)
        if text:
            reasoning_parts.append(text)
    if not reasoning_parts:
        return False

    moved_text = "\n".join(reasoning_parts)
    content = message.get("content")
    if content is None:
        message["content"] = moved_text
    elif isinstance(content, str):
        message["content"] = f"{moved_text}{content}"
    elif isinstance(content, list):
        message["content"] = [{"type": "text", "text": moved_text}] + content
    else:
        message["content"] = moved_text
    return True


def _normalize_no_thinking_response_payload(
    path: str, request_payload: Any, response_payload: Any
) -> tuple[Any, bool]:
    if path != "v1/chat/completions" or _request_enables_thinking(request_payload):
        return response_payload, False
    if not isinstance(response_payload, dict):
        return response_payload, False
    choices = response_payload.get("choices")
    if not isinstance(choices, list):
        return response_payload, False

    changed = False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            changed = _move_reasoning_to_content_for_no_thinking(message) or changed
    return response_payload, changed


def _force_thinking_enabled() -> bool:
    return _env_bool_any(
        ("VLLM_JETSPEC_FORCE_THINKING", "VLLM_DFLASH_FORCE_THINKING"), False
    )


def _default_enable_thinking() -> bool:
    return _env_bool_any(
        (
            "VLLM_JETSPEC_DEFAULT_ENABLE_THINKING",
            "VLLM_DFLASH_DEFAULT_ENABLE_THINKING",
        ),
        False,
    )


def _force_thinking_system_prompt() -> str:
    return (
        _env_first(
            "VLLM_JETSPEC_FORCE_THINKING_SYSTEM_PROMPT",
            "VLLM_DFLASH_FORCE_THINKING_SYSTEM_PROMPT",
            default="",
        )
        or ""
    ).strip()


def _force_thinking_marker() -> str:
    return (
        _env_first(
            "VLLM_JETSPEC_FORCE_THINKING_MARKER",
            "VLLM_DFLASH_FORCE_THINKING_MARKER",
            default="",
        )
        or ""
    ).strip()


def _content_contains_text(content: Any, text: str) -> bool:
    if not text:
        return True
    if isinstance(content, str):
        return text in content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and text in str(item.get("text") or ""):
                return True
            if isinstance(item, str) and text in item:
                return True
    return False


def _append_text_to_content(content: Any, text: str) -> Any:
    if not text:
        return content
    if isinstance(content, str):
        cleaned = content.replace("/no_think", "").replace("/nothink", "").rstrip()
        if text in cleaned:
            return cleaned
        return f"{cleaned} {text}".strip()
    if isinstance(content, list):
        cleaned_items: list[Any] = []
        has_text = False
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                updated = dict(item)
                updated["text"] = (
                    updated["text"].replace("/no_think", "").replace("/nothink", "")
                )
                if text in updated["text"]:
                    has_text = True
                cleaned_items.append(updated)
            elif isinstance(item, str):
                updated = item.replace("/no_think", "").replace("/nothink", "")
                if text in updated:
                    has_text = True
                cleaned_items.append(updated)
            else:
                cleaned_items.append(item)
        if has_text:
            return cleaned_items
        return cleaned_items + [{"type": "text", "text": text}]
    return content


def _force_thinking_messages(messages: list[Any]) -> list[Any]:
    normalized_messages = [
        dict(message) if isinstance(message, dict) else message for message in messages
    ]
    system_prompt = _force_thinking_system_prompt()
    if system_prompt and not any(
        isinstance(message, dict)
        and message.get("role") == "system"
        and _content_contains_text(message.get("content"), system_prompt)
        for message in normalized_messages
    ):
        insert_at = 0
        while (
            insert_at < len(normalized_messages)
            and isinstance(normalized_messages[insert_at], dict)
            and normalized_messages[insert_at].get("role") in {"system", "developer"}
        ):
            insert_at += 1
        normalized_messages.insert(
            insert_at, {"role": "system", "content": system_prompt}
        )

    marker = _force_thinking_marker()
    if marker:
        for index in range(len(normalized_messages) - 1, -1, -1):
            message = normalized_messages[index]
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            updated = dict(message)
            updated["content"] = _append_text_to_content(
                updated.get("content"), marker
            )
            normalized_messages[index] = updated
            break
    return normalized_messages


def _normalize_mistral_message_roles(messages: list[Any]) -> list[Any]:
    normalized_messages: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized_messages.append(message)
            continue
        role = message.get("role")
        content = message.get("content")
        normalized_content = _normalize_mistral_message_content(content)
        if role not in MISTRAL_ROLE_ALIASES and normalized_content is content:
            normalized_messages.append(message)
            continue
        normalized = dict(message)
        if role in MISTRAL_ROLE_ALIASES:
            normalized["role"] = MISTRAL_ROLE_ALIASES[role]
        if normalized_content is not content:
            normalized["content"] = normalized_content
        normalized_messages.append(normalized)
    return normalized_messages


def _normalize_mistral_message_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    text_parts: list[str] = []
    for item in content:
        if (
            not isinstance(item, dict)
            or item.get("type") not in MISTRAL_TEXT_CONTENT_TYPES
        ):
            return content
        text_parts.append(str(item.get("text") or ""))
    return "".join(text_parts)


def _normalize_inference_request_body(
    path: str, body: bytes, request_payload: Any, runtime_settings: Any | None = None
) -> tuple[bytes, Any]:
    if (
        path not in INFERENCE_PATHS
        or not isinstance(request_payload, dict)
    ):
        return body, request_payload
    has_messages = isinstance(request_payload.get("messages"), list)
    has_responses_input_messages = path == "v1/responses" and isinstance(
        request_payload.get("input"), list
    )
    if not has_messages and not has_responses_input_messages:
        return body, request_payload
    force_thinking = _force_thinking_enabled()
    default_enable_thinking = _default_enable_thinking()
    normalized = dict(request_payload)
    if _vllm_uses_mistral_tokenizer(runtime_settings):
        for key in CHAT_TEMPLATE_REQUEST_KEYS:
            normalized.pop(key, None)
        if has_messages:
            normalized["messages"] = _normalize_mistral_message_roles(
                request_payload["messages"]
            )
        if has_responses_input_messages:
            normalized["input"] = _normalize_mistral_message_roles(
                request_payload["input"]
            )
    else:
        if not has_messages:
            return body, request_payload
        raw_kwargs = normalized.get("chat_template_kwargs")
        chat_template_kwargs = dict(raw_kwargs) if isinstance(raw_kwargs, dict) else {}
        if force_thinking and chat_template_kwargs.get("enable_thinking") is not True:
            chat_template_kwargs["enable_thinking"] = True
        elif "enable_thinking" not in chat_template_kwargs:
            chat_template_kwargs["enable_thinking"] = default_enable_thinking
        if chat_template_kwargs != raw_kwargs:
            normalized["chat_template_kwargs"] = chat_template_kwargs
    if force_thinking and has_messages:
        normalized["messages"] = _force_thinking_messages(request_payload["messages"])
    if normalized == request_payload:
        return body, request_payload
    return json.dumps(normalized, separators=(",", ":")).encode("utf-8"), normalized


def _route_active_lora_request_body(
    body: bytes, request_payload: Any
) -> tuple[bytes, Any, bool]:
    if not _live_lora_proxy_enabled() or not isinstance(request_payload, dict):
        return body, request_payload, False
    active = _read_active_live_lora_adapter(settings.live_lora_checkpoint_dir)
    adapter_name = active.get("adapter_name")
    if not adapter_name:
        return body, request_payload, False
    requested_model = str(request_payload.get("model") or settings.served_model_name)
    if requested_model not in {settings.served_model_name, settings.model_name}:
        return body, request_payload, False
    if requested_model == adapter_name:
        return body, request_payload, False
    routed = dict(request_payload)
    routed["model"] = str(adapter_name)
    return json.dumps(routed, separators=(",", ":")).encode("utf-8"), routed, True


def _configure_logging() -> None:
    logging.basicConfig(
        level=getattr(
            logging,
            (
                _env_first(
                    "VLLM_JETSPEC_LOG_LEVEL", "VLLM_DFLASH_LOG_LEVEL", default="INFO"
                )
                or "INFO"
            ).upper(),
            logging.INFO,
        ),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def _count_jsonl_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except FileNotFoundError:
        return 0
    except OSError:
        return 0


def _unlink_runtime_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        logger.exception("Failed to remove non-persistent runtime file %s", path)


@dataclass(frozen=True)
class TrafficSnapshot:
    rows: int
    size_bytes: int
    mtime_ns: int

    def as_dict(self) -> dict[str, int]:
        return {
            "rows": self.rows,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
        }


def _traffic_snapshot(path: Path) -> TrafficSnapshot:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return TrafficSnapshot(rows=0, size_bytes=0, mtime_ns=0)
    except OSError:
        return TrafficSnapshot(rows=0, size_bytes=0, mtime_ns=0)
    return TrafficSnapshot(
        rows=_count_jsonl_rows(path),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def _effective_training_snapshot(
    training_data_path: Path, traffic_path: Path
) -> TrafficSnapshot:
    training_snapshot = _traffic_snapshot(training_data_path)
    if not _mix_live_traffic_enabled() or training_data_path == traffic_path:
        return training_snapshot
    traffic_snapshot = _traffic_snapshot(traffic_path)
    return TrafficSnapshot(
        rows=training_snapshot.rows + traffic_snapshot.rows,
        size_bytes=training_snapshot.size_bytes + traffic_snapshot.size_bytes,
        mtime_ns=max(training_snapshot.mtime_ns, traffic_snapshot.mtime_ns),
    )


def _mix_live_traffic_enabled() -> bool:
    return _env_bool_any(
        ("VLLM_JETSPEC_MIX_TRAFFIC_DATA", "VLLM_DFLASH_MIX_TRAFFIC_DATA"),
        True,
    )


def _training_state_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "training_state.json"


def _active_live_lora_adapter_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "active_adapter.json"


def _read_active_live_lora_adapter(checkpoint_dir: Path) -> dict[str, Any]:
    payload = _read_json_file(_active_live_lora_adapter_path(checkpoint_dir))
    if not payload:
        return {}
    adapter_path = payload.get("adapter_path")
    if not adapter_path:
        return {}
    if not Path(str(adapter_path)).exists():
        return {}
    return payload


def _write_active_live_lora_adapter(
    checkpoint_dir: Path, payload: dict[str, Any]
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _active_live_lora_adapter_path(checkpoint_dir)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _read_last_trained_traffic_snapshot(checkpoint_dir: Path) -> dict[str, int] | None:
    try:
        payload = json.loads(
            _training_state_path(checkpoint_dir).read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    snapshot = payload.get("last_trained_traffic_snapshot")
    if not isinstance(snapshot, dict):
        return None
    try:
        return {
            "rows": int(snapshot.get("rows", 0)),
            "size_bytes": int(snapshot.get("size_bytes", 0)),
            "mtime_ns": int(snapshot.get("mtime_ns", 0)),
        }
    except (TypeError, ValueError):
        return None


def _write_last_trained_traffic_snapshot(
    checkpoint_dir: Path,
    snapshot: TrafficSnapshot,
    *,
    run_id: str,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_trained_traffic_snapshot": snapshot.as_dict(),
        "last_trained_run_id": run_id,
        "updated_at": time.time(),
    }
    state_path = _training_state_path(checkpoint_dir)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(state_path)


def _latest_checkpoint_step(checkpoint_dir: Path) -> int | None:
    latest = _read_json_file(checkpoint_dir / "latest.json")
    if not latest:
        return None
    try:
        return int(latest.get("step") or 0)
    except (TypeError, ValueError):
        return None


def _traffic_snapshot_matches(
    snapshot: TrafficSnapshot,
    trained_snapshot: dict[str, int] | None,
) -> bool:
    return trained_snapshot == snapshot.as_dict()


def _tail_jsonl(path: Path, limit: int = 100) -> list[dict[str, Any]]:
    rows: deque[dict[str, Any]] = deque(maxlen=max(1, limit))
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except FileNotFoundError:
        return []
    except OSError:
        return []
    return list(rows)


_LIVE_TRAINING_DATA_LOCK = threading.Lock()


def _prepare_live_training_rows(payload: Any, task: str) -> list[dict[str, Any]]:
    samples = _extract_training_samples(payload)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        row = dict(sample)
        row["task"] = task
        row.setdefault("id", str(uuid.uuid4()))
        row.setdefault("ts", time.time())
        error = _validate_live_training_row(row)
        if error:
            row["_error"] = error
        rows.append(row)
    invalid = [row.get("_error") for row in rows if row.get("_error")]
    if invalid:
        raise ValueError("; ".join(str(error) for error in invalid[:5]))
    return rows


def _extract_training_samples(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("samples"), list):
            return payload["samples"]
        if isinstance(payload.get("data"), list):
            return payload["data"]
        return [payload]
    return []


def _validate_live_training_row(row: dict[str, Any]) -> str | None:
    task = row.get("task")
    if task == "sft":
        if isinstance(row.get("messages"), list) and row["messages"]:
            return None
        if row.get("text"):
            return None
        if row.get("prompt") is not None and row.get("completion") is not None:
            return None
        return "SFT rows need messages, text, or prompt+completion"
    if task == "dpo":
        has_prompt = row.get("prompt") is not None or isinstance(row.get("messages"), list)
        if has_prompt and row.get("chosen") is not None and row.get("rejected") is not None:
            return None
        return "DPO rows need prompt/messages plus chosen and rejected"
    return f"Unsupported live training task: {task}"


def _append_live_training_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with _LIVE_TRAINING_DATA_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                row = {key: value for key, value in row.items() if key != "_error"}
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _live_training_counts(path: Path) -> dict[str, int]:
    counts = {"sft": 0, "dpo": 0, "total": 0}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                task = row.get("task")
                if task in {"sft", "dpo"}:
                    counts[task] += 1
                    counts["total"] += 1
    except FileNotFoundError:
        return counts
    except OSError:
        return counts
    return counts


def _summarize_training_metrics(path: Path) -> dict[str, Any]:
    rows = _tail_jsonl(path, limit=250)
    if not rows:
        return {"available": False}

    latest = rows[-1]
    run_id = latest.get("run_id")
    if run_id:
        run_rows = [row for row in rows if row.get("run_id") == run_id]
    else:
        run_rows = rows
    step_rows = [row for row in run_rows if row.get("event") == "step"]
    latest_step = step_rows[-1] if step_rows else None
    complete_rows = [row for row in run_rows if row.get("event") == "run_complete"]
    latest_complete = complete_rows[-1] if complete_rows else None

    summary: dict[str, Any] = {
        "available": True,
        "latest_event": latest.get("event"),
        "latest_ts": latest.get("ts"),
        "run_id": run_id,
        "events_in_tail": len(run_rows),
    }
    if latest_step:
        summary.update(
            {
                "step": latest_step.get("step"),
                "local_step": latest_step.get("local_step"),
                "max_steps": latest_step.get("max_steps"),
                "progress": latest_step.get("progress"),
                "loss": latest_step.get("loss"),
                "elapsed_seconds": latest_step.get("elapsed_seconds"),
            }
        )
    if latest_complete:
        summary["last_complete"] = {
            "step": latest_complete.get("step"),
            "completed_steps": latest_complete.get("completed_steps"),
            "max_steps": latest_complete.get("max_steps"),
            "loss": latest_complete.get("loss"),
            "stopped": latest_complete.get("stopped"),
            "elapsed_seconds": latest_complete.get("elapsed_seconds"),
            "ts": latest_complete.get("ts"),
        }
    return summary


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {"payload": payload}


def _parse_prometheus_samples(text: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_SAMPLE_RE.match(line)
        if not match:
            continue
        metric = match.group(1)
        labels = match.group(2)
        if not any(
            keyword in metric.lower() for keyword in SPECULATIVE_METRIC_KEYWORDS
        ):
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        sample: dict[str, Any] = {"metric": metric, "value": value}
        if labels:
            sample["labels"] = labels[1:-1]
        samples.append(sample)
    return samples


def _summarize_vllm_speculative_metrics(text: str) -> dict[str, Any]:
    samples = _parse_prometheus_samples(text)
    if not samples:
        return {"available": False, "samples": []}

    acceptance_rates = [
        sample
        for sample in samples
        if "accept" in sample["metric"].lower() and "rate" in sample["metric"].lower()
    ]
    accepted_tokens = [
        sample
        for sample in samples
        if "accept" in sample["metric"].lower()
        and "token" in sample["metric"].lower()
        and "rate" not in sample["metric"].lower()
        and _is_counter_value_sample(sample)
    ]
    draft_tokens = [
        sample
        for sample in samples
        if ("draft" in sample["metric"].lower() or "spec" in sample["metric"].lower())
        and "token" in sample["metric"].lower()
        and "accept" not in sample["metric"].lower()
        and _is_counter_value_sample(sample)
    ]
    draft_counts = [
        sample
        for sample in samples
        if "num_drafts" in sample["metric"].lower()
        and "token" not in sample["metric"].lower()
        and _is_counter_value_sample(sample)
    ]
    accepted_token_totals = _prefer_aggregate_token_samples(accepted_tokens)
    draft_token_totals = _prefer_aggregate_token_samples(draft_tokens)
    draft_count_totals = _prefer_aggregate_token_samples(draft_counts)

    summary: dict[str, Any] = {
        "available": True,
        "sample_count": len(samples),
        "samples": samples[:50],
    }
    if acceptance_rates:
        summary["reported_acceptance_rates"] = acceptance_rates[:10]
        summary["acceptance_rate"] = acceptance_rates[-1]["value"]
    if accepted_token_totals:
        summary["accepted_token_total"] = sum(
            sample["value"] for sample in accepted_token_totals
        )
    if draft_token_totals:
        summary["draft_token_total"] = sum(
            sample["value"] for sample in draft_token_totals
        )
    if draft_count_totals:
        summary["draft_count_total"] = sum(
            sample["value"] for sample in draft_count_totals
        )
    if "accepted_token_total" in summary and "draft_token_total" in summary:
        denominator = float(summary["draft_token_total"])
        if denominator > 0:
            summary["computed_acceptance_rate"] = (
                float(summary["accepted_token_total"]) / denominator
            )
    if "accepted_token_total" in summary and "draft_count_total" in summary:
        denominator = float(summary["draft_count_total"])
        if denominator > 0:
            summary["mean_acceptance_length"] = (
                float(summary["accepted_token_total"]) / denominator
            )
    return summary


def _is_counter_value_sample(sample: dict[str, Any]) -> bool:
    metric = str(sample.get("metric") or "").lower()
    return not metric.endswith(("_bucket", "_created", "_sum", "_count"))


def _prefer_aggregate_token_samples(
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregate_samples = [
        sample
        for sample in samples
        if "per_pos" not in str(sample.get("metric") or "").lower()
        and "position=" not in str(sample.get("labels") or "")
    ]
    return aggregate_samples or samples


def _accepted_tokens_by_position(
    vllm_metrics: dict[str, Any] | None,
) -> dict[str, float]:
    positions: dict[str, float] = {}
    if not vllm_metrics:
        return positions
    for sample in vllm_metrics.get("samples") or []:
        metric = str(sample.get("metric") or "").lower()
        labels = str(sample.get("labels") or "")
        if (
            "accepted_tokens_per_pos" not in metric
            or "position=" not in labels
            or not _is_counter_value_sample(sample)
        ):
            continue
        match = re.search(r'position="?(?P<position>\d+)"?', labels)
        if not match:
            continue
        positions[match.group("position")] = float(sample.get("value") or 0.0)
    return positions


def _effective_accept_depth(positions: dict[str, float]) -> int:
    accepted_positions = [
        int(position) for position, value in positions.items() if float(value) > 0.0
    ]
    if not accepted_positions:
        return 0
    return max(accepted_positions) + 1


def _speculation_policy_summary(
    settings: Settings,
    *,
    vllm_metrics: dict[str, Any] | None = None,
    proposer_stats: dict[str, Any] | None = None,
    suffix_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = settings.speculative_config or {}
    effective_method = str(config.get("method") or "none")
    requested_method = str(getattr(settings, "spec_method", "") or "")
    automatic = requested_method in AUTO_SPEC_METHODS
    serving_path = {
        "suffix": "native_suffix",
        "custom_class": "plugin_hybrid_suffix_jetspec",
        "dflash": "native_jetspec",
        "none": "disabled",
    }.get(effective_method, effective_method or "disabled")
    uses_suffix = effective_method in {"suffix", "custom_class"}
    uses_plugin_hybrid = effective_method == "custom_class"
    uses_jetspec = uses_plugin_hybrid or serving_path == "native_jetspec"
    plugin_available = bool(_read_json_file(settings.checkpoint_dir / "latest.json"))

    summary: dict[str, Any] = {
        "requested_method": requested_method,
        "effective_method": effective_method,
        "serving_path": serving_path,
        "automatic": automatic,
        "uses_suffix_decoding": uses_suffix,
        "uses_jetspec": uses_jetspec,
        "uses_plugin_hybrid": uses_plugin_hybrid,
        "plugin_checkpoint_available": plugin_available,
        "num_speculative_tokens": config.get("num_speculative_tokens"),
    }
    if automatic:
        if serving_path == "native_suffix":
            summary["selection_reason"] = (
                "auto selected native suffix because it is the fastest current "
                "plugin-supervised serving path for this model"
            )
        elif serving_path == "plugin_hybrid_suffix_jetspec":
            summary["selection_reason"] = (
                "auto policy selected the custom plugin hybrid proposer"
            )
        else:
            summary["selection_reason"] = "auto selected the configured native draft path"

    if suffix_cache:
        summary["suffix_cache_backend"] = suffix_cache.get("suffix_cache_backend")
        summary["suffix_cache_requests"] = suffix_cache.get("suffix_cache_requests")
        summary["suffix_cache_suffixes"] = suffix_cache.get("suffix_cache_suffixes")

    if proposer_stats:
        suffix_share = float(proposer_stats.get("suffix_share_of_proposed_tokens") or 0.0)
        jetspec_share = float(
            proposer_stats.get("jetspec_share_of_proposed_tokens") or 0.0
        )
        summary["plugin_proposal_fill_rate"] = proposer_stats.get("proposal_fill_rate")
        summary["plugin_suffix_share"] = suffix_share
        summary["plugin_jetspec_share"] = jetspec_share
        summary["plugin_min_adapter_tokens"] = proposer_stats.get(
            "min_adapter_tokens"
        )
        summary["plugin_max_adapter_tokens"] = proposer_stats.get(
            "max_adapter_tokens"
        )
        summary["plugin_average_adapter_budget_tokens"] = proposer_stats.get(
            "average_adapter_budget_tokens_per_sequence"
        )
        summary["plugin_jetspec_min_token_prob"] = proposer_stats.get(
            "jetspec_min_token_prob"
        )
        summary["plugin_jetspec_filtered_tokens"] = proposer_stats.get(
            "jetspec_filtered_tokens"
        )
        summary["plugin_dominant_draft_source"] = (
            "jetspec" if jetspec_share > suffix_share else "suffix"
        )

    if vllm_metrics:
        positions = _accepted_tokens_by_position(vllm_metrics)
        summary["accepted_token_total"] = vllm_metrics.get("accepted_token_total")
        summary["draft_token_total"] = vllm_metrics.get("draft_token_total")
        summary["draft_count_total"] = vllm_metrics.get("draft_count_total")
        summary["acceptance_rate"] = vllm_metrics.get("computed_acceptance_rate")
        summary["mean_acceptance_length"] = vllm_metrics.get(
            "mean_acceptance_length"
        )
        summary["effective_accept_depth"] = _effective_accept_depth(positions)
        summary["accepted_tokens_by_position"] = positions
    return summary


settings = Settings.from_env()
vllm_process = VllmProcess(settings)
traffic_recorder = TrafficRecorder(settings)
coordinator = TrainingCoordinator(settings, vllm_process)
_background_tasks: set[asyncio.Task[Any]] = set()


def _schedule_traffic_record(
    endpoint: str, request_payload: Any, response_payload: Any
) -> None:
    task = asyncio.create_task(
        traffic_recorder.record(endpoint, request_payload, response_payload)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    task.add_done_callback(_log_background_task_exception)


def _log_background_task_exception(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("Background task failed")


app = FastAPI(title="vLLM JIT JetSpec Supervisor")


@app.on_event("startup")
async def startup() -> None:
    _configure_logging()
    draft_resolution = settings.jetspec_draft_model_resolution
    logger.info(
        "Starting vLLM JIT JetSpec supervisor served_model=%s spec_method=%s spec_config=%s draft_resolution=%s speculation_policy=%s",
        settings.served_model_name,
        settings.spec_method,
        settings.speculative_config,
        draft_resolution.as_dict() if draft_resolution else None,
        _speculation_policy_summary(settings),
    )
    await vllm_process.start()
    coordinator.start()


@app.on_event("shutdown")
async def shutdown() -> None:
    await coordinator.stop()
    if vllm_process.process is not None and vllm_process.process.poll() is None:
        vllm_process.process.terminate()


@app.get("/health")
async def health() -> Response:
    vllm_status = vllm_process.status()
    payload = {
        "ok": True,
        "model": settings.model_name,
        "served_model_name": settings.served_model_name,
        "vllm_base_url": settings.vllm_base_url,
        "vllm_process": vllm_status,
        "spec_method": settings.spec_method,
        "speculative_config": settings.speculative_config,
        "speculation_policy": _speculation_policy_summary(settings),
        "jetspec_draft_model_resolution": (
            settings.jetspec_draft_model_resolution.as_dict()
            if settings.jetspec_draft_model_resolution
            else None
        ),
    }
    if not vllm_status["running"]:
        payload["ok"] = False
        return Response(
            content=json.dumps(payload),
            status_code=503,
            media_type="application/json",
        )
    return Response(
        content=json.dumps(payload),
        status_code=200,
        media_type="application/json",
    )


@app.get("/jit/status")
async def jit_status() -> dict[str, Any]:
    status = await coordinator.status()
    status["vllm_sleeping"] = await vllm_process.is_sleeping()
    vllm_metrics = await vllm_process.speculative_metrics()
    suffix_cache = traffic_recorder.status()
    status["vllm_speculative_metrics"] = vllm_metrics
    status["suffix_cache"] = suffix_cache
    status["background_traffic_tasks"] = len(_background_tasks)
    status["speculation_policy"] = _speculation_policy_summary(
        settings,
        vllm_metrics=vllm_metrics,
        proposer_stats=status.get("proposer_stats"),
        suffix_cache=suffix_cache,
    )
    return status


@app.get("/jit/training_metrics")
async def jit_training_metrics(limit: int = 100) -> dict[str, Any]:
    limit = min(max(1, limit), 1000)
    return {
        "metrics_path": str(settings.metrics_path),
        "summary": _summarize_training_metrics(settings.metrics_path),
        "rows": _tail_jsonl(settings.metrics_path, limit=limit),
    }


@app.get("/jit/train/data")
async def jit_train_data(limit: int = 100) -> dict[str, Any]:
    limit = min(max(1, limit), 1000)
    path = settings.live_training_data_path
    return {
        "path": str(path),
        "snapshot": _traffic_snapshot(path).as_dict(),
        "counts": _live_training_counts(path),
        "rows": _tail_jsonl(path, limit=limit),
        "latest_live_lora_checkpoint": _read_json_file(
            settings.live_lora_checkpoint_dir / "latest.json"
        ),
        "active_live_lora_adapter": _read_active_live_lora_adapter(
            settings.live_lora_checkpoint_dir
        ),
    }


@app.post("/jit/train/sft")
async def jit_train_sft(
    request: Request,
) -> dict[str, Any]:
    return await _ingest_live_training_rows(
        request,
        task="sft",
    )


@app.post("/jit/train/dpo")
async def jit_train_dpo(
    request: Request,
) -> dict[str, Any]:
    return await _ingest_live_training_rows(
        request,
        task="dpo",
    )


async def _ingest_live_training_rows(
    request: Request,
    *,
    task: str,
) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Expected JSON body") from exc
    try:
        rows = _prepare_live_training_rows(payload, task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _append_live_training_rows(settings.live_training_data_path, rows)
    return {
        "status": "accepted",
        "task": task,
        "accepted": len(rows),
        "path": str(settings.live_training_data_path),
        "snapshot": _traffic_snapshot(settings.live_training_data_path).as_dict(),
        "counts": _live_training_counts(settings.live_training_data_path),
    }


@app.get("/jit/speculation_metrics")
async def jit_speculation_metrics() -> dict[str, Any]:
    vllm_metrics = await vllm_process.speculative_metrics()
    proposer_stats = _read_json_file(settings.proposer_stats_path)
    suffix_cache = traffic_recorder.status()
    return {
        "policy": _speculation_policy_summary(
            settings,
            vllm_metrics=vllm_metrics,
            proposer_stats=proposer_stats,
            suffix_cache=suffix_cache,
        ),
        "vllm": vllm_metrics,
        "hybrid_proposer": proposer_stats,
        "suffix_cache": suffix_cache,
        "proposer_stats_path": str(settings.proposer_stats_path),
    }


@app.post("/jit/train_once")
async def jit_train_once(wait: bool = False, force: bool = False) -> dict[str, Any]:
    return await jit_train_jetspec(wait=wait, force=force)


@app.post("/jit/train/jetspec")
async def jit_train_jetspec(wait: bool = False, force: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    if wait:
        returncode = await coordinator.train_jetspec_once(force=force)
        payload = {
            "status": "completed" if returncode in (0, None) else "failed",
            "trainer": "jetspec",
            "seconds": time.monotonic() - started,
            "trainer_returncode": returncode,
            "training_metrics": _summarize_training_metrics(settings.metrics_path),
            "latest_checkpoint": _read_json_file(
                settings.checkpoint_dir / "latest.json"
            ),
        }
        if returncode not in (0, None):
            raise HTTPException(status_code=500, detail=payload)
        return payload
    asyncio.create_task(coordinator.train_jetspec_once(force=force))
    return {"status": "scheduled", "trainer": "jetspec", "force": force}


@app.post("/jit/checkpoints/sync")
async def jit_sync_checkpoints(
    direction: str = "both", force: bool = False
) -> dict[str, Any]:
    direction = direction.strip().lower()
    if direction not in {"both", "download", "upload"}:
        raise HTTPException(
            status_code=400, detail="direction must be one of both, download, upload"
        )
    payload: dict[str, Any] = {"status": "completed", "direction": direction}
    if direction in {"both", "download"}:
        payload["download"] = await coordinator._download_hub_checkpoint(
            trigger="api", force=force
        )
    if direction in {"both", "upload"}:
        payload["upload"] = await coordinator._upload_hub_checkpoint(trigger="api")
    payload["latest_checkpoint"] = _read_json_file(
        settings.checkpoint_dir / "latest.json"
    )
    return payload


@app.post("/jit/train/live_lora")
async def jit_train_live_lora(
    wait: bool = False,
    force: bool = False,
    task_mix: str | None = None,
    max_steps: int | None = None,
    max_seq_len: int | None = None,
    loss_vocab_sample_size: int | None = None,
    lora_target_modules: str | None = None,
) -> dict[str, Any]:
    if max_steps is not None and max_steps < 0:
        raise HTTPException(status_code=400, detail="max_steps must be >= 0")
    if task_mix is not None:
        task_mix = task_mix.strip().lower()
        if task_mix not in {"mixed", "sft", "dpo"}:
            raise HTTPException(
                status_code=400,
                detail="task_mix must be one of: mixed, sft, dpo",
            )
    if max_seq_len is not None and max_seq_len < 1:
        raise HTTPException(status_code=400, detail="max_seq_len must be >= 1")
    if loss_vocab_sample_size is not None and loss_vocab_sample_size < 0:
        raise HTTPException(
            status_code=400,
            detail="loss_vocab_sample_size must be >= 0",
        )
    if lora_target_modules is not None:
        lora_target_modules = lora_target_modules.strip()
        if not lora_target_modules:
            raise HTTPException(
                status_code=400,
                detail="lora_target_modules must not be empty",
            )
    started = time.monotonic()
    if wait:
        returncode = await coordinator.train_live_lora_once(
            force=force,
            task_mix=task_mix,
            max_steps=max_steps,
            max_seq_len=max_seq_len,
            loss_vocab_sample_size=loss_vocab_sample_size,
            lora_target_modules=lora_target_modules,
        )
        payload = {
            "status": "completed" if returncode in (0, None) else "failed",
            "trainer": "live_lora",
            "task_mix": task_mix,
            "max_steps": max_steps,
            "max_seq_len": max_seq_len,
            "loss_vocab_sample_size": loss_vocab_sample_size,
            "lora_target_modules": lora_target_modules,
            "seconds": time.monotonic() - started,
            "trainer_returncode": returncode,
            "training_metrics": _summarize_training_metrics(settings.metrics_path),
            "latest_live_lora_checkpoint": _read_json_file(
                settings.live_lora_checkpoint_dir / "latest.json"
            ),
            "active_live_lora_adapter": _read_active_live_lora_adapter(
                settings.live_lora_checkpoint_dir
            ),
        }
        if returncode not in (0, None):
            raise HTTPException(status_code=500, detail=payload)
        return payload
    asyncio.create_task(
        coordinator.train_live_lora_once(
            force=force,
            task_mix=task_mix,
            max_steps=max_steps,
            max_seq_len=max_seq_len,
            loss_vocab_sample_size=loss_vocab_sample_size,
            lora_target_modules=lora_target_modules,
        )
    )
    return {
        "status": "scheduled",
        "trainer": "live_lora",
        "force": force,
        "task_mix": task_mix,
        "max_steps": max_steps,
        "max_seq_len": max_seq_len,
        "loss_vocab_sample_size": loss_vocab_sample_size,
        "lora_target_modules": lora_target_modules,
    }


@app.post("/jit/rebuild_suffix_cache")
async def jit_rebuild_suffix_cache(limit: int = 500) -> dict[str, Any]:
    limit = min(max(1, limit), 5000)
    return await traffic_recorder.rebuild_suffix_cache(limit=limit)


@app.post("/jit/sleep")
async def jit_sleep(level: int | None = None) -> dict[str, Any]:
    sleep_level = settings.sleep_level if level is None else int(level)
    checkpoint_sync = await coordinator._download_hub_checkpoint(
        trigger="before_sleep"
    )
    await vllm_process.sleep(sleep_level)
    return {"status": "sleeping", "level": sleep_level, "checkpoint_sync": checkpoint_sync}


@app.post("/jit/wake")
async def jit_wake() -> dict[str, str]:
    await coordinator.enter_inference()
    await coordinator.exit_inference()
    return {"status": "awake"}


def _training_in_progress_response() -> Response:
    return Response(
        content=json.dumps(
            {
                "error": {
                    "message": "Training is running; retry through the manager when the model is awake.",
                    "type": "training_in_progress",
                }
            }
        ),
        status_code=503,
        headers={"Retry-After": "5"},
        media_type="application/json",
    )


def _model_wake_failed_response(exc: Exception) -> Response:
    return Response(
        content=json.dumps(
            {
                "error": {
                    "message": "vLLM did not wake up in time; retry the request or call /jit/wake.",
                    "type": "model_wake_failed",
                    "detail": str(exc),
                }
            }
        ),
        status_code=503,
        headers={"Retry-After": "5"},
        media_type="application/json",
    )


def _vllm_upstream_unavailable_response(exc: Exception) -> Response:
    return Response(
        content=json.dumps(
            {
                "error": {
                    "message": "vLLM upstream is unavailable; the supervisor is running but the vLLM engine is not accepting connections.",
                    "type": "vllm_upstream_unavailable",
                    "detail": str(exc),
                    "vllm_process": vllm_process.status(),
                }
            }
        ),
        status_code=503,
        headers={"Retry-After": "5"},
        media_type="application/json",
    )


@app.api_route(
    "/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
)
async def proxy(path: str, request: Request):
    body = await request.body()
    request_payload = _parse_json(body)
    is_inference = path in INFERENCE_PATHS
    lora_routed = False
    if is_inference:
        if await coordinator.trainer_running():
            return _training_in_progress_response()
        body, request_payload = _normalize_inference_request_body(
            path, body, request_payload, settings
        )
        body, request_payload, lora_routed = _route_active_lora_request_body(
            body, request_payload
        )
    if is_inference:
        try:
            await coordinator.enter_inference()
        except Exception as exc:
            logger.warning("Failed to wake vLLM for inference: %s", exc)
            await coordinator.exit_inference()
            return _model_wake_failed_response(exc)
    client = httpx.AsyncClient(timeout=None)
    upstream_request = client.build_request(
        request.method,
        f"{settings.vllm_base_url}/{path}",
        params=request.query_params,
        headers=_filtered_request_headers(request),
        content=body,
    )
    try:
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        if is_inference:
            await coordinator.exit_inference()
        logger.warning("vLLM upstream request failed: %s", exc)
        return _vllm_upstream_unavailable_response(exc)
    except Exception:
        await client.aclose()
        if is_inference:
            await coordinator.exit_inference()
        raise

    content_type = upstream_response.headers.get("content-type", "")
    request_streaming = bool(
        isinstance(request_payload, dict) and request_payload.get("stream")
    )
    if "text/event-stream" in content_type or request_streaming:
        return StreamingResponse(
            _stream_response(
                upstream_response, client, is_inference, path, request_payload
            ),
            status_code=upstream_response.status_code,
            headers=_filtered_response_headers(upstream_response),
            media_type=content_type or None,
        )

    try:
        content = await upstream_response.aread()
        response_payload = _parse_json(content)
        if is_inference:
            response_payload, changed = _normalize_no_thinking_response_payload(
                path, request_payload, response_payload
            )
            if (
                lora_routed
                and isinstance(response_payload, dict)
                and response_payload.get("model") != settings.served_model_name
            ):
                response_payload = dict(response_payload)
                response_payload["model"] = settings.served_model_name
                changed = True
            if changed:
                content = json.dumps(response_payload).encode("utf-8")
            _schedule_traffic_record(path, request_payload, response_payload)
    finally:
        await upstream_response.aclose()
        await client.aclose()
        if is_inference:
            await coordinator.exit_inference()
    response_headers = _filtered_response_headers(upstream_response)
    if is_inference and changed:
        response_headers.pop("content-length", None)
    return Response(
        content=content,
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=content_type or None,
    )


async def _stream_response(
    upstream_response: httpx.Response,
    client: httpx.AsyncClient,
    is_inference: bool,
    path: str = "",
    request_payload: Any = None,
):
    capture = StreamingCompletionCapture() if is_inference else None
    try:
        async for chunk in upstream_response.aiter_bytes():
            if capture is not None:
                capture.feed(chunk)
            yield chunk
    finally:
        await upstream_response.aclose()
        await client.aclose()
        if is_inference:
            if capture is not None:
                capture.finish()
                completion = capture.completion
                if completion:
                    _schedule_traffic_record(
                        path, request_payload, _completion_payload(completion)
                    )
            await coordinator.exit_inference()


def _parse_json(body: bytes) -> Any:
    if not body:
        return None
    try:
        return json.loads(body)
    except Exception:
        return None


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.public_host, port=settings.public_port)


if __name__ == "__main__":
    main()
