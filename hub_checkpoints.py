from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from .paths import model_checkpoint_slug


CHECKPOINT_PATTERNS = ("checkpoint-step-*.pt",)


@dataclass(frozen=True)
class HubCheckpointConfig:
    repo_id: str
    repo_type: str
    revision: str
    path_prefix: str
    download_enabled: bool
    upload_enabled: bool
    create_repo: bool
    private: bool
    sync_interval_seconds: float
    sync_on_inference: bool

    @property
    def enabled(self) -> bool:
        return bool(self.repo_id and (self.download_enabled or self.upload_enabled))

    def path_in_repo(self, name: str) -> str:
        clean_name = str(name).lstrip("/")
        if not self.path_prefix:
            return clean_name
        return f"{self.path_prefix.rstrip('/')}/{clean_name}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "revision": self.revision,
            "path_prefix": self.path_prefix,
            "download_enabled": self.download_enabled,
            "upload_enabled": self.upload_enabled,
            "create_repo": self.create_repo,
            "private": self.private,
            "sync_interval_seconds": self.sync_interval_seconds,
            "sync_on_inference": self.sync_on_inference,
        }


def config_from_env(
    model_name: str, served_model_name: str | None = None
) -> HubCheckpointConfig | None:
    repo_id = _env_first(
        "VLLM_JETSPEC_HF_CHECKPOINT_REPO",
        "VLLM_DFLASH_HF_CHECKPOINT_REPO",
    )
    if not repo_id:
        return None
    path_prefix = _env_first(
        "VLLM_JETSPEC_HF_CHECKPOINT_PATH_PREFIX",
        "VLLM_DFLASH_HF_CHECKPOINT_PATH_PREFIX",
        default=model_checkpoint_slug(served_model_name or model_name),
    )
    return HubCheckpointConfig(
        repo_id=repo_id,
        repo_type=_env_first(
            "VLLM_JETSPEC_HF_CHECKPOINT_REPO_TYPE",
            "VLLM_DFLASH_HF_CHECKPOINT_REPO_TYPE",
            default="model",
        )
        or "model",
        revision=_env_first(
            "VLLM_JETSPEC_HF_CHECKPOINT_REVISION",
            "VLLM_DFLASH_HF_CHECKPOINT_REVISION",
            default="main",
        )
        or "main",
        path_prefix=(path_prefix or "").strip("/"),
        download_enabled=_env_bool_any(
            (
                "VLLM_JETSPEC_HF_CHECKPOINT_DOWNLOAD",
                "VLLM_DFLASH_HF_CHECKPOINT_DOWNLOAD",
            ),
            True,
        ),
        upload_enabled=_env_bool_any(
            (
                "VLLM_JETSPEC_HF_CHECKPOINT_UPLOAD",
                "VLLM_DFLASH_HF_CHECKPOINT_UPLOAD",
            ),
            True,
        ),
        create_repo=_env_bool_any(
            (
                "VLLM_JETSPEC_HF_CHECKPOINT_CREATE_REPO",
                "VLLM_DFLASH_HF_CHECKPOINT_CREATE_REPO",
            ),
            False,
        ),
        private=_env_bool_any(
            (
                "VLLM_JETSPEC_HF_CHECKPOINT_REPO_PRIVATE",
                "VLLM_DFLASH_HF_CHECKPOINT_REPO_PRIVATE",
            ),
            True,
        ),
        sync_interval_seconds=max(
            0.0,
            _env_float_any(
                (
                    "VLLM_JETSPEC_HF_CHECKPOINT_SYNC_INTERVAL_SECONDS",
                    "VLLM_DFLASH_HF_CHECKPOINT_SYNC_INTERVAL_SECONDS",
                ),
                300.0,
            ),
        ),
        sync_on_inference=_env_bool_any(
            (
                "VLLM_JETSPEC_HF_CHECKPOINT_SYNC_ON_INFERENCE",
                "VLLM_DFLASH_HF_CHECKPOINT_SYNC_ON_INFERENCE",
            ),
            True,
        ),
    )


def status_payload(
    model_name: str,
    checkpoint_dir: Path,
    *,
    served_model_name: str | None = None,
    last_sync: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config_from_env(model_name, served_model_name=served_model_name)
    return {
        "config": config.as_dict() if config else {"enabled": False},
        "latest_checkpoint": _read_json(checkpoint_dir / "latest.json"),
        "last_sync": last_sync or {},
    }


def download_newer_checkpoint(
    *,
    model_name: str,
    checkpoint_dir: Path,
    config: HubCheckpointConfig | None = None,
) -> dict[str, Any]:
    config = config or config_from_env(model_name)
    if config is None or not config.enabled or not config.download_enabled:
        return {"status": "disabled", "direction": "download"}
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    local_latest = _read_json(checkpoint_dir / "latest.json")
    local_step = _checkpoint_step(local_latest)
    try:
        remote_latest_path = _hf_hub_download(
            repo_id=config.repo_id,
            filename=config.path_in_repo("latest.json"),
            repo_type=config.repo_type,
            revision=config.revision,
        )
        remote_latest = _read_json(Path(remote_latest_path))
    except Exception as exc:
        return {
            "status": "failed",
            "direction": "download",
            "error": str(exc),
            "repo_id": config.repo_id,
            "path": config.path_in_repo("latest.json"),
        }
    remote_step = _checkpoint_step(remote_latest)
    if remote_step <= local_step:
        return {
            "status": "up_to_date",
            "direction": "download",
            "local_step": local_step,
            "remote_step": remote_step,
        }
    checkpoint_name = _checkpoint_name_from_latest(remote_latest)
    if not checkpoint_name:
        return {
            "status": "failed",
            "direction": "download",
            "error": "remote latest.json has no checkpoint pointer",
            "remote_step": remote_step,
        }
    try:
        remote_checkpoint_path = _hf_hub_download(
            repo_id=config.repo_id,
            filename=config.path_in_repo(checkpoint_name),
            repo_type=config.repo_type,
            revision=config.revision,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "direction": "download",
            "error": str(exc),
            "remote_step": remote_step,
            "checkpoint": checkpoint_name,
        }
    local_checkpoint = checkpoint_dir / checkpoint_name
    _atomic_copy(Path(remote_checkpoint_path), local_checkpoint)
    local_payload = _downloaded_latest_payload(
        remote_latest,
        checkpoint_name=checkpoint_name,
        checkpoint_path=local_checkpoint,
        config=config,
    )
    _atomic_write_json(checkpoint_dir / "latest.json", local_payload)
    return {
        "status": "downloaded",
        "direction": "download",
        "local_step": local_step,
        "remote_step": remote_step,
        "checkpoint_path": str(local_checkpoint),
    }


def upload_latest_checkpoint(
    *,
    model_name: str,
    checkpoint_dir: Path,
    config: HubCheckpointConfig | None = None,
) -> dict[str, Any]:
    config = config or config_from_env(model_name)
    if config is None or not config.enabled or not config.upload_enabled:
        return {"status": "disabled", "direction": "upload"}
    latest_path = checkpoint_dir / "latest.json"
    latest = _read_json(latest_path)
    checkpoint_path = _checkpoint_path_from_latest(latest_path, latest)
    if checkpoint_path is None or not checkpoint_path.is_file():
        return {
            "status": "skipped",
            "direction": "upload",
            "reason": "no_local_checkpoint",
        }
    try:
        api = _hf_api()
        if config.create_repo:
            api.create_repo(
                repo_id=config.repo_id,
                repo_type=config.repo_type,
                private=config.private,
                exist_ok=True,
            )
        checkpoint_name = checkpoint_path.name
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=config.path_in_repo(checkpoint_name),
            repo_id=config.repo_id,
            repo_type=config.repo_type,
            revision=config.revision,
            commit_message=f"Upload JetSpec checkpoint step {_checkpoint_step(latest)}",
        )
        remote_latest = _remote_latest_payload(
            latest,
            checkpoint_name=checkpoint_name,
            config=config,
            timestamp_key="uploaded_at",
        )
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            json.dump(remote_latest, handle, indent=2)
            handle.write("\n")
            temp_latest = Path(handle.name)
        try:
            api.upload_file(
                path_or_fileobj=str(temp_latest),
                path_in_repo=config.path_in_repo("latest.json"),
                repo_id=config.repo_id,
                repo_type=config.repo_type,
                revision=config.revision,
                commit_message=f"Promote JetSpec checkpoint step {_checkpoint_step(latest)}",
            )
        finally:
            temp_latest.unlink(missing_ok=True)
    except Exception as exc:
        return {
            "status": "failed",
            "direction": "upload",
            "error": str(exc),
            "repo_id": config.repo_id,
            "path_prefix": config.path_prefix,
        }
    return {
        "status": "uploaded",
        "direction": "upload",
        "step": _checkpoint_step(latest),
        "checkpoint": checkpoint_path.name,
        "repo_id": config.repo_id,
        "path_prefix": config.path_prefix,
    }


def _checkpoint_path_from_latest(latest_path: Path, latest: dict[str, Any]) -> Path | None:
    checkpoint = latest.get("checkpoint_path") or latest.get("checkpoint")
    if not checkpoint:
        return None
    path = Path(str(checkpoint))
    if not path.is_absolute():
        path = latest_path.parent / path
    return path


def _remote_latest_payload(
    latest: dict[str, Any],
    *,
    checkpoint_name: str,
    config: HubCheckpointConfig,
    timestamp_key: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checkpoint": checkpoint_name,
        "checkpoint_path": checkpoint_name,
        "step": _checkpoint_step(latest),
        "saved_at": latest.get("saved_at"),
        "model_name": latest.get("model_name"),
        "num_speculative_tokens": latest.get("num_speculative_tokens"),
        "quality": _sanitize_quality(latest.get("quality")),
        "hub_repo_id": config.repo_id,
        "hub_path_prefix": config.path_prefix,
        "hub_revision": config.revision,
        timestamp_key: time.time(),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _downloaded_latest_payload(
    latest: dict[str, Any],
    *,
    checkpoint_name: str,
    checkpoint_path: Path,
    config: HubCheckpointConfig,
) -> dict[str, Any]:
    payload = _remote_latest_payload(
        latest,
        checkpoint_name=checkpoint_name,
        config=config,
        timestamp_key="downloaded_at",
    )
    payload["checkpoint_path"] = str(checkpoint_path)
    return payload


def _sanitize_quality(value: Any) -> dict[str, int | float | bool]:
    if not isinstance(value, dict):
        return {}
    clean: dict[str, int | float | bool] = {}
    for key, metric in value.items():
        if not isinstance(key, str):
            continue
        if isinstance(metric, bool):
            clean[key] = metric
        elif isinstance(metric, (int, float)):
            clean[key] = metric
    return clean


def _checkpoint_name_from_latest(latest: dict[str, Any]) -> str:
    checkpoint = latest.get("checkpoint") or latest.get("checkpoint_path")
    if not checkpoint:
        return ""
    return Path(str(checkpoint)).name


def _checkpoint_step(payload: dict[str, Any]) -> int:
    try:
        return int(payload.get("step") or 0)
    except (TypeError, ValueError):
        return 0


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, tmp)
    tmp.replace(target)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw.strip() != "":
            return raw.strip()
    return default


def _env_bool_any(names: tuple[str, ...], default: bool) -> bool:
    for name in names:
        raw = os.getenv(name)
        if raw is not None:
            return raw.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _env_float_any(names: tuple[str, ...], default: float) -> float:
    raw = _env_first(*names)
    if raw is None:
        return default
    return float(raw)


def _hf_api() -> Any:
    from huggingface_hub import HfApi

    return HfApi()


def _hf_hub_download(**kwargs: Any) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill promoted JetSpec checkpoints to Hugging Face Hub."
    )
    parser.add_argument("--self-check", action="store_true", help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command")
    upload = subparsers.add_parser(
        "upload", help="upload checkpoint-dir/latest.json and its checkpoint"
    )
    upload.add_argument("--model-name", required=True)
    upload.add_argument("--served-model-name")
    upload.add_argument("--checkpoint-dir", required=True, type=Path)
    return parser


def _upload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = config_from_env(args.model_name, served_model_name=args.served_model_name)
    return upload_latest_checkpoint(
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        config=config,
    )


def _self_check() -> None:
    parser = _build_parser()
    try:
        parser.parse_args(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    old_env = os.environ.copy()
    try:
        os.environ.clear()
        os.environ.update(
            {
                "VLLM_JETSPEC_HF_CHECKPOINT_REPO": "namespace/private-spec-checkpoints",
            }
        )
        config = config_from_env("base/model", served_model_name="smolagent")
        assert config is not None
        assert config.private is True
        assert config.path_prefix == "smolagent"
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.self_check:
        _self_check()
        return 0
    if args.command == "upload":
        result = _upload_from_args(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1 if result.get("status") == "failed" else 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
