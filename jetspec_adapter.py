from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JetSpecAdapterConfig:
    vocab_size: int
    hidden_size: int = 256
    num_speculative_tokens: int = 8
    pad_token_id: int = 0
    architecture: str = "pooled_gru"

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if self.num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1")
        normalized = str(self.architecture or "").strip().lower()
        if normalized not in {"pooled_gru", "autoregressive_gru"}:
            raise ValueError(
                "architecture must be one of: pooled_gru, autoregressive_gru"
            )


def _torch():
    import torch

    return torch


class JetSpecAdapter:
    """Small causal draft adapter used for JIT JetSpec experiments.

    Native JetSpec serving uses a Hugging Face draft-head checkpoint through the
    JetSpec vLLM fork. This adapter is the lightweight local-training path used
    by the custom proposer while collecting traffic and iterating quickly.
    """

    def __new__(cls, config: JetSpecAdapterConfig):  # type: ignore[override]
        torch = _torch()
        architecture = str(config.architecture or "pooled_gru").strip().lower()

        if architecture == "autoregressive_gru":

            class _AutoregressiveJetSpecAdapter(torch.nn.Module):
                def __init__(self, adapter_config: JetSpecAdapterConfig):
                    super().__init__()
                    self.adapter_config = adapter_config
                    self.embedding = torch.nn.Embedding(
                        adapter_config.vocab_size,
                        adapter_config.hidden_size,
                        padding_idx=adapter_config.pad_token_id,
                    )
                    self.encoder = torch.nn.GRU(
                        input_size=adapter_config.hidden_size,
                        hidden_size=adapter_config.hidden_size,
                        batch_first=True,
                    )
                    self.decoder_cell = torch.nn.GRUCell(
                        input_size=adapter_config.hidden_size,
                        hidden_size=adapter_config.hidden_size,
                    )
                    self.future_offsets = torch.nn.Embedding(
                        adapter_config.num_speculative_tokens,
                        adapter_config.hidden_size,
                    )
                    self.norm = torch.nn.LayerNorm(adapter_config.hidden_size)
                    self.output_bias = torch.nn.Parameter(
                        torch.zeros(adapter_config.vocab_size)
                    )

                def forward(self, input_ids, labels=None):
                    embedded = self.embedding(input_ids)
                    _outputs, hidden = self.encoder(embedded)
                    state = hidden[-1]
                    logits_by_position = []
                    for position in range(self.adapter_config.num_speculative_tokens):
                        offset = self.future_offsets.weight[position].unsqueeze(0)
                        logits = (
                            self.norm(state + offset) @ self.embedding.weight.T
                        ) + self.output_bias
                        logits_by_position.append(logits)
                        if labels is not None and position < labels.shape[1]:
                            next_token_ids = labels[:, position]
                            predicted = logits.argmax(dim=-1)
                            next_token_ids = torch.where(
                                next_token_ids == self.adapter_config.pad_token_id,
                                predicted,
                                next_token_ids,
                            )
                        else:
                            next_token_ids = logits.argmax(dim=-1)
                        state = self.decoder_cell(
                            self.embedding(next_token_ids),
                            state,
                        )
                    return torch.stack(logits_by_position, dim=1)

            return _AutoregressiveJetSpecAdapter(config)

        class _TorchJetSpecAdapter(torch.nn.Module):
            def __init__(self, adapter_config: JetSpecAdapterConfig):
                super().__init__()
                self.adapter_config = adapter_config
                self.embedding = torch.nn.Embedding(
                    adapter_config.vocab_size,
                    adapter_config.hidden_size,
                    padding_idx=adapter_config.pad_token_id,
                )
                self.encoder = torch.nn.GRU(
                    input_size=adapter_config.hidden_size,
                    hidden_size=adapter_config.hidden_size,
                    batch_first=True,
                )
                self.norm = torch.nn.LayerNorm(adapter_config.hidden_size)
                self.head = torch.nn.Linear(
                    adapter_config.hidden_size,
                    adapter_config.num_speculative_tokens * adapter_config.vocab_size,
                )

            def forward(self, input_ids):
                embedded = self.embedding(input_ids)
                _outputs, hidden = self.encoder(embedded)
                pooled = self.norm(hidden[-1])
                logits = self.head(pooled)
                return logits.view(
                    input_ids.shape[0],
                    self.adapter_config.num_speculative_tokens,
                    self.adapter_config.vocab_size,
                )

        return _TorchJetSpecAdapter(config)


def save_checkpoint(
    path: str | os.PathLike[str],
    model: Any,
    *,
    optimizer: Any | None = None,
    step: int = 0,
    metadata: dict[str, Any] | None = None,
) -> None:
    torch = _torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    config = getattr(model, "adapter_config", None)
    payload: dict[str, Any] = {
        "version": 2,
        "family": "jetspec",
        "step": int(step),
        "config": asdict(config) if config is not None else None,
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(target)


def load_checkpoint(
    path: str | os.PathLike[str], *, map_location: str = "cpu"
) -> dict[str, Any]:
    torch = _torch()
    payload = torch.load(Path(path), map_location=map_location)
    if payload.get("version") not in {1, 2}:
        raise ValueError(
            f"Unsupported JetSpec adapter checkpoint version: {payload.get('version')}"
        )
    return payload


def load_adapter(path: str | os.PathLike[str], *, map_location: str = "cpu"):
    payload = load_checkpoint(path, map_location=map_location)
    config_payload = payload.get("config")
    if not config_payload:
        raise ValueError("Checkpoint does not include adapter config")
    model = JetSpecAdapter(JetSpecAdapterConfig(**config_payload))
    model.load_state_dict(payload["model_state_dict"])
    model.to(map_location)
    model.eval()
    return model, payload
