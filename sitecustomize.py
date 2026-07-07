from __future__ import annotations

import os
from pathlib import Path
import sys


def _running_pip() -> bool:
    return Path(sys.argv[0]).name.lower().startswith("pip")


def _torch_float32_matmul_precision() -> str:
    return (
        os.getenv("TORCH_FLOAT32_MATMUL_PRECISION")
        or os.getenv("VLLM_JETSPEC_TORCH_FLOAT32_MATMUL_PRECISION")
        or "high"
    ).strip().lower()


def _configure_torch_defaults() -> None:
    if _running_pip():
        return
    precision = _torch_float32_matmul_precision()
    if precision in {"", "0", "false", "off", "none"}:
        return
    if precision not in {"highest", "high", "medium"}:
        raise ValueError(
            "TORCH_FLOAT32_MATMUL_PRECISION must be one of 'highest', 'high', "
            "'medium', or 'off'."
        )

    import torch

    torch.set_float32_matmul_precision(precision)


_configure_torch_defaults()
