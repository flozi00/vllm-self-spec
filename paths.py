from __future__ import annotations

from dataclasses import dataclass
import os
import re
from pathlib import Path


HF_MODEL_WEIGHT_GLOBS = (
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "pytorch_model.bin.index.json",
)

JETSPEC_PRETRAINED_DRAFT_HEADS = (
    ("qwen3-8b", "JetSpec/jetspec-qwen3-8b"),
    ("qwen3-30b-a3b", "JetSpec/jetspec-qwen3-30b-a3b"),
    ("qwen3.6-35b-a3b", "JetSpec/jetspec-Qwen3.6-35B-A3B"),
    ("gemma4-26b-a4b0-it", "JetSpec/jetspec-gemma4-26B-A4B0-it"),
    ("gemma-4-26b-a4b0-it", "JetSpec/jetspec-gemma4-26B-A4B0-it"),
    ("gpt-oss-20b", "JetSpec/jetspec-gpt-oss-20b"),
    ("step-3.7-flash", "JetSpec/jetspec-Step-3.7-Flash"),
    ("step3p7-flash", "JetSpec/jetspec-Step-3.7-Flash"),
)


@dataclass(frozen=True)
class DraftModelResolution:
    model: str
    source: str
    local_path: Path | None = None
    checked_local_paths: tuple[Path, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "source": self.source,
            "local_path": str(self.local_path) if self.local_path else "",
            "checked_local_paths": [str(path) for path in self.checked_local_paths],
        }


def model_checkpoint_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "--", model_name.strip())
    slug = slug.strip("-._")
    return slug or "unknown-model"


def checkpoint_dir_for_model(root: str | os.PathLike[str], model_name: str) -> Path:
    return Path(root) / model_checkpoint_slug(model_name)


def checkpoint_dir_from_env(model_name: str, data_dir: str | os.PathLike[str]) -> Path:
    explicit_dir = _env_first_nonempty(
        "VLLM_JETSPEC_CHECKPOINT_DIR",
        "VLLM_DFLASH_CHECKPOINT_DIR",
    )
    if explicit_dir:
        return Path(explicit_dir)
    root = Path(
        _env_first_nonempty(
            "VLLM_JETSPEC_CHECKPOINT_ROOT",
            "VLLM_DFLASH_CHECKPOINT_ROOT",
        )
        or str(Path(data_dir) / "checkpoints")
    )
    return checkpoint_dir_for_model(root, model_name)


def latest_checkpoint_json_from_env(
    model_name: str, data_dir: str | os.PathLike[str]
) -> Path:
    explicit_latest = _env_first_nonempty(
        "VLLM_JETSPEC_LATEST_CHECKPOINT_JSON",
        "VLLM_DFLASH_LATEST_CHECKPOINT_JSON",
    )
    if explicit_latest:
        return Path(explicit_latest)
    return checkpoint_dir_from_env(model_name, data_dir) / "latest.json"


def native_jetspec_draft_model_resolution_from_env(
    model_name: str,
    data_dir: str | os.PathLike[str],
) -> DraftModelResolution:
    explicit_model = _env_first_nonempty(
        "VLLM_JETSPEC_DRAFT_MODEL",
        "VLLM_JETSPEC_DRAFT_HEAD",
        "JETSPEC_DRAFT_HEAD",
    )
    if explicit_model:
        return DraftModelResolution(model=explicit_model, source="explicit")

    checked_local_paths: list[Path] = []
    for path in _native_jetspec_local_candidates(model_name, data_dir):
        checked_local_paths.append(path)
        if _looks_like_hf_model_dir(path):
            return DraftModelResolution(
                model=str(path),
                source="local",
                local_path=path,
                checked_local_paths=tuple(checked_local_paths),
            )

    pretrained_model = _env_first_nonempty(
        "VLLM_JETSPEC_PRETRAINED_DRAFT_MODEL",
        "VLLM_JETSPEC_PRETRAINED_DRAFT_HEAD",
        "VLLM_JETSPEC_PRETRAINED_SPECULATOR",
    )
    if not pretrained_model:
        pretrained_model = pretrained_jetspec_draft_model_for_target(model_name)
    if pretrained_model:
        return DraftModelResolution(
            model=pretrained_model,
            source="pretrained",
            checked_local_paths=tuple(checked_local_paths),
        )

    return DraftModelResolution(
        model="",
        source="missing",
        checked_local_paths=tuple(checked_local_paths),
    )


def _native_jetspec_local_candidates(
    model_name: str,
    data_dir: str | os.PathLike[str],
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for name in (
        "VLLM_JETSPEC_DRAFT_MODEL_DIR",
        "VLLM_JETSPEC_DRAFT_HEAD_DIR",
        "VLLM_JETSPEC_NATIVE_DRAFT_MODEL_DIR",
        "JETSPEC_DRAFT_HEAD_DIR",
    ):
        raw = _env_nonempty(name)
        if raw:
            candidates.append(Path(raw))
    checkpoint_dir = checkpoint_dir_from_env(model_name, data_dir)
    candidates.extend(
        [
            checkpoint_dir / "jetspec",
            checkpoint_dir / "draft_head",
            checkpoint_dir / "draft_model",
            checkpoint_dir,
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return tuple(deduped)


def _looks_like_hf_model_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return any(
        next(path.glob(pattern), None) is not None for pattern in HF_MODEL_WEIGHT_GLOBS
    )


def _env_nonempty(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip()


def _env_first_nonempty(*names: str) -> str | None:
    for name in names:
        value = _env_nonempty(name)
        if value:
            return value
    return None


def pretrained_jetspec_draft_model_for_target(model_name: str) -> str:
    normalized = model_name.lower().replace("_", "-")
    for needle, draft_head in JETSPEC_PRETRAINED_DRAFT_HEADS:
        if needle in normalized:
            return draft_head
    return ""
