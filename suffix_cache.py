from __future__ import annotations

import json
import os
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    from ._vllm_dflash_jit_rust import (
        RustSuffixCache as _RustSuffixCache,
        filter_jetspec_drafts as _rust_filter_jetspec_drafts,
        merge_and_summarize_hybrid_drafts as _rust_merge_and_summarize_hybrid_drafts,
        merge_hybrid_drafts as _rust_merge_hybrid_drafts,
        summarize_hybrid_counts as _rust_summarize_hybrid_counts,
    )
except Exception:
    try:
        from _vllm_dflash_jit_rust import (
            RustSuffixCache as _RustSuffixCache,
            filter_jetspec_drafts as _rust_filter_jetspec_drafts,
            merge_and_summarize_hybrid_drafts as _rust_merge_and_summarize_hybrid_drafts,
            merge_hybrid_drafts as _rust_merge_hybrid_drafts,
            summarize_hybrid_counts as _rust_summarize_hybrid_counts,
        )
    except Exception:
        _RustSuffixCache = None
        _rust_filter_jetspec_drafts = None
        _rust_merge_and_summarize_hybrid_drafts = None
        _rust_merge_hybrid_drafts = None
        _rust_summarize_hybrid_counts = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class SuffixCacheConfig:
    max_tree_depth: int = 24
    max_cached_requests: int = 10_000
    max_spec_factor: float = 1.0
    min_token_prob: float = 0.1
    min_match_tokens: int = 1

    def __post_init__(self) -> None:
        if self.max_tree_depth < 1:
            raise ValueError("max_tree_depth must be >= 1")
        if self.max_cached_requests < 0:
            raise ValueError("max_cached_requests must be >= 0")
        if self.max_spec_factor <= 0:
            raise ValueError("max_spec_factor must be > 0")
        if not 0 <= self.min_token_prob <= 1:
            raise ValueError("min_token_prob must be in [0, 1]")
        if self.min_match_tokens < 1:
            raise ValueError("min_match_tokens must be >= 1")


@dataclass(frozen=True)
class SuffixCacheRecord:
    tokens: list[int]
    min_next_index: int = 1


class CpuSuffixCache:
    """Frequency suffix cache that keeps all state in Python CPU memory.

    vLLM's built-in suffix proposer uses Arctic Inference internally. This
    cache is intentionally separate and persistent so the JIT trainer/proxy can
    checkpoint traffic patterns and expose a simple custom proposer path.
    """

    def __init__(self, config: SuffixCacheConfig | None = None):
        self.config = config or SuffixCacheConfig()
        self._requests: OrderedDict[str, SuffixCacheRecord] = OrderedDict()
        self._next_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self._native = self._new_native_backend()

    @property
    def backend_name(self) -> str:
        return "rust" if self._native is not None else "python"

    @property
    def request_count(self) -> int:
        return len(self._requests)

    @property
    def suffix_count(self) -> int:
        if self._native is not None:
            return int(self._native.suffix_count())
        return len(self._next_counts)

    def add_sequence(
        self,
        token_ids: Iterable[int],
        request_id: str | None = None,
        *,
        min_next_index: int = 1,
    ) -> None:
        tokens = [int(token_id) for token_id in token_ids]
        if len(tokens) < 2:
            return
        min_next_index = self._normalize_min_next_index(tokens, min_next_index)
        if request_id is None:
            request_id = f"request-{len(self._requests) + 1}"
        if request_id in self._requests:
            old_record = self._requests.pop(request_id)
            self._remove_from_index(old_record.tokens, old_record.min_next_index)
        record = SuffixCacheRecord(tokens=tokens, min_next_index=min_next_index)
        self._requests[request_id] = record
        self._add_to_index(record.tokens, record.min_next_index)
        self._evict_if_needed()

    def propose(self, context_token_ids: Iterable[int], max_tokens: int) -> list[int]:
        context = [int(token_id) for token_id in context_token_ids]
        if max_tokens <= 0 or not context:
            return []
        if self._native is not None:
            proposal = [
                int(token_id) for token_id in self._native.propose(context, max_tokens)
            ]
            return self._with_in_context_suffix(context, proposal, max_tokens)

        match = self._longest_suffix_match(context)
        proposed: list[int] = []
        if match:
            max_by_factor = max(1, int(len(match) * self.config.max_spec_factor))
            budget = min(max_tokens, max_by_factor)

            rolling_context = list(context)
            for _ in range(budget):
                suffix = self._longest_suffix_match(rolling_context)
                if not suffix:
                    break
                next_token = self._best_next_token(suffix)
                if next_token is None:
                    break
                proposed.append(next_token)
                rolling_context.append(next_token)
        return self._with_in_context_suffix(context, proposed, max_tokens)

    def propose_many(
        self, context_batches: Iterable[Iterable[int]], max_tokens: int
    ) -> list[list[int]]:
        contexts = [
            [int(token_id) for token_id in context_token_ids]
            for context_token_ids in context_batches
        ]
        if max_tokens <= 0:
            return [[] for _context in contexts]
        native_propose_many = getattr(self._native, "propose_many", None)
        if native_propose_many is not None:
            proposals = [
                [int(token_id) for token_id in proposal]
                for proposal in native_propose_many(contexts, max_tokens)
            ]
            return [
                self._with_in_context_suffix(context, proposal, max_tokens)
                for context, proposal in zip(contexts, proposals)
            ]
        return [self.propose(context, max_tokens) for context in contexts]

    def propose_many_with_adapter_inputs(
        self,
        context_batches: Iterable[Iterable[int]],
        suffix_max_tokens: int,
        max_adapter_tokens: int,
    ) -> tuple[list[list[int]], list[list[int]], list[int]]:
        contexts = [
            [int(token_id) for token_id in context_token_ids]
            for context_token_ids in context_batches
        ]
        native_fused = getattr(self._native, "propose_many_with_adapter_inputs", None)
        if native_fused is not None:
            suffixes, adapter_contexts, adapter_budgets = native_fused(
                contexts, max(0, int(suffix_max_tokens)), max(0, int(max_adapter_tokens))
            )
            suffix_lists = [
                self._with_in_context_suffix(
                    context,
                    [int(token_id) for token_id in proposal],
                    suffix_max_tokens,
                )
                for context, proposal in zip(contexts, suffixes)
            ]
            return (
                suffix_lists,
                [
                    context + suffix_tokens
                    for context, suffix_tokens in zip(contexts, suffix_lists)
                ],
                [int(budget) for budget in adapter_budgets],
            )

        suffixes = self.propose_many(contexts, suffix_max_tokens)
        adapter_budget = max(0, int(max_adapter_tokens))
        adapter_contexts: list[list[int]] = []
        adapter_budgets: list[int] = []
        for context, suffix_tokens in zip(contexts, suffixes):
            adapter_contexts.append(context + suffix_tokens)
            adapter_budgets.append(adapter_budget if context else 0)
        return suffixes, adapter_contexts, adapter_budgets

    def save(self, path: str | os.PathLike[str]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "config": asdict(self.config),
            "requests": [
                [
                    request_id,
                    {
                        "tokens": record.tokens,
                        "min_next_index": record.min_next_index,
                    },
                ]
                for request_id, record in self._requests.items()
            ],
        }
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp_path.replace(target)

    @classmethod
    def load(cls, path: str | os.PathLike[str], config: SuffixCacheConfig | None = None) -> "CpuSuffixCache":
        source = Path(path)
        if not source.exists():
            return cls(config)
        payload = json.loads(source.read_text(encoding="utf-8"))
        loaded_config = SuffixCacheConfig(**payload.get("config", {}))
        cache = cls(config or loaded_config)
        for request_id, record_payload in payload.get("requests", []):
            token_ids, min_next_index = cls._parse_record_payload(record_payload)
            cache.add_sequence(
                token_ids,
                request_id=str(request_id),
                min_next_index=min_next_index,
            )
        return cache

    @staticmethod
    def _parse_record_payload(record_payload) -> tuple[list[int], int]:
        if isinstance(record_payload, dict):
            token_ids = record_payload.get("tokens", [])
            min_next_index = int(record_payload.get("min_next_index", 1) or 1)
        else:
            token_ids = record_payload
            min_next_index = 1
        return [int(token_id) for token_id in token_ids], min_next_index

    @staticmethod
    def _normalize_min_next_index(tokens: list[int], min_next_index: int) -> int:
        if len(tokens) < 2:
            return 1
        return min(max(1, int(min_next_index)), len(tokens))

    def _add_to_index(self, tokens: list[int], min_next_index: int) -> None:
        if self._native is not None:
            self._native.add_sequence_with_min_next_index(tokens, min_next_index)
            return
        max_depth = self.config.max_tree_depth
        min_depth = self.config.min_match_tokens
        if max_depth < min_depth:
            return
        for next_index in range(min_next_index, len(tokens)):
            next_token = tokens[next_index]
            start = max(0, next_index - max_depth)
            stop = next_index - min_depth + 1
            if stop <= start:
                continue
            for suffix_start in range(start, stop):
                suffix = tuple(tokens[suffix_start:next_index])
                self._next_counts[suffix][next_token] += 1

    def _remove_from_index(self, tokens: list[int], min_next_index: int) -> None:
        if self._native is not None:
            self._native.remove_sequence_with_min_next_index(tokens, min_next_index)
            return
        max_depth = self.config.max_tree_depth
        min_depth = self.config.min_match_tokens
        if max_depth < min_depth:
            return
        empty_suffixes: list[tuple[int, ...]] = []
        for next_index in range(min_next_index, len(tokens)):
            next_token = tokens[next_index]
            start = max(0, next_index - max_depth)
            stop = next_index - min_depth + 1
            if stop <= start:
                continue
            for suffix_start in range(start, stop):
                suffix = tuple(tokens[suffix_start:next_index])
                counts = self._next_counts.get(suffix)
                if not counts:
                    continue
                counts[next_token] -= 1
                if counts[next_token] <= 0:
                    del counts[next_token]
                if not counts:
                    empty_suffixes.append(suffix)
        for suffix in empty_suffixes:
            self._next_counts.pop(suffix, None)

    def _evict_if_needed(self) -> None:
        if self.config.max_cached_requests == 0:
            self._requests.clear()
            if self._native is not None:
                self._native.clear()
            else:
                self._next_counts.clear()
            return
        while len(self._requests) > self.config.max_cached_requests:
            _request_id, record = self._requests.popitem(last=False)
            self._remove_from_index(record.tokens, record.min_next_index)

    def _longest_suffix_match(self, context: list[int]) -> tuple[int, ...]:
        max_depth = min(len(context), self.config.max_tree_depth)
        min_depth = self.config.min_match_tokens
        if max_depth < min_depth:
            return ()
        for size in range(max_depth, min_depth - 1, -1):
            suffix = tuple(context[-size:])
            if suffix in self._next_counts:
                return suffix
        return ()

    def _best_next_token(self, suffix: tuple[int, ...]) -> int | None:
        counts = self._next_counts.get(suffix)
        if not counts:
            return None
        token_id, count = counts.most_common(1)[0]
        total = sum(counts.values())
        if total <= 0 or count / total < self.config.min_token_prob:
            return None
        return int(token_id)

    def _with_in_context_suffix(
        self, context: list[int], proposal: list[int], max_tokens: int
    ) -> list[int]:
        if (
            len(proposal) >= max_tokens
            or not _env_bool("VLLM_JETSPEC_IN_CONTEXT_SUFFIX", True)
        ):
            return proposal[:max_tokens]
        if not proposal and not _env_bool(
            "VLLM_JETSPEC_IN_CONTEXT_SUFFIX_BOOTSTRAP", True
        ):
            return proposal[:max_tokens]
        extra = self._propose_from_in_context_repeats(
            context + proposal, max_tokens - len(proposal)
        )
        if not extra:
            return proposal[:max_tokens]
        return (proposal + extra)[:max_tokens]

    def _propose_from_in_context_repeats(
        self, context: list[int], max_tokens: int
    ) -> list[int]:
        min_match_tokens = _env_int(
            "VLLM_JETSPEC_IN_CONTEXT_SUFFIX_MIN_MATCH",
            self.config.min_match_tokens,
        )
        if max_tokens <= 0 or len(context) <= min_match_tokens:
            return []
        window = _env_int("VLLM_JETSPEC_IN_CONTEXT_SUFFIX_WINDOW", 8192)
        if window > 0 and len(context) > window:
            rolling = list(context[-window:])
        else:
            rolling = list(context)
        proposed: list[int] = []
        for _ in range(max_tokens):
            next_token = self._best_in_context_next_token(
                rolling, min_match_tokens=min_match_tokens
            )
            if next_token is None:
                break
            proposed.append(next_token)
            rolling.append(next_token)
            if window > 0 and len(rolling) > window:
                rolling = rolling[-window:]
        return proposed

    def _best_in_context_next_token(
        self, context: list[int], *, min_match_tokens: int
    ) -> int | None:
        max_depth = min(len(context) - 1, self.config.max_tree_depth)
        min_depth = max(1, int(min_match_tokens))
        if max_depth < min_depth:
            return None
        for size in range(max_depth, min_depth - 1, -1):
            suffix = context[-size:]
            counts: Counter[int] = Counter()
            for offset in range(0, len(context) - size):
                if context[offset : offset + size] == suffix:
                    counts[context[offset + size]] += 1
            if not counts:
                continue
            token_id, count = counts.most_common(1)[0]
            total = sum(counts.values())
            if total > 0 and count / total >= self.config.min_token_prob:
                return int(token_id)
        return None

    def _new_native_backend(self):
        backend = (
            os.getenv("VLLM_JETSPEC_SUFFIX_BACKEND")
            or os.getenv("VLLM_DFLASH_SUFFIX_BACKEND")
            or "auto"
        ).strip().lower()
        if backend in {"python", "py", "off", "disabled", "false", "0"}:
            return None
        if _RustSuffixCache is None:
            return None
        native = _RustSuffixCache(
            int(self.config.max_tree_depth),
            int(self.config.max_cached_requests),
            float(self.config.max_spec_factor),
            float(self.config.min_token_prob),
            int(self.config.min_match_tokens),
        )
        if not hasattr(native, "add_sequence_with_min_next_index"):
            return None
        return native


def merge_hybrid_drafts(
    suffix_batches: Iterable[Iterable[int]],
    jetspec_batches: Iterable[Iterable[int]],
    num_speculative_tokens: int,
) -> tuple[list[list[int]], list[int], list[int]]:
    suffixes = [
        [int(token_id) for token_id in suffix_tokens]
        for suffix_tokens in suffix_batches
    ]
    jetspecs = [
        [int(token_id) for token_id in jetspec_tokens]
        for jetspec_tokens in jetspec_batches
    ]
    if _rust_merge_hybrid_drafts is not None:
        proposals, suffix_counts, jetspec_counts = _rust_merge_hybrid_drafts(
            suffixes, jetspecs, int(num_speculative_tokens)
        )
        return (
            [[int(token_id) for token_id in proposal] for proposal in proposals],
            [int(count) for count in suffix_counts],
            [int(count) for count in jetspec_counts],
        )

    batch_count = max(len(suffixes), len(jetspecs))
    proposals: list[list[int]] = []
    suffix_counts: list[int] = []
    jetspec_counts: list[int] = []
    budget = max(0, int(num_speculative_tokens))
    for index in range(batch_count):
        suffix_tokens = suffixes[index] if index < len(suffixes) else []
        jetspec_tokens = jetspecs[index] if index < len(jetspecs) else []
        suffix_take = min(len(suffix_tokens), budget)
        jetspec_take = min(len(jetspec_tokens), max(0, budget - suffix_take))
        proposal = suffix_tokens[:suffix_take] + jetspec_tokens[:jetspec_take]
        proposals.append(proposal)
        suffix_counts.append(suffix_take)
        jetspec_counts.append(jetspec_take)
    return proposals, suffix_counts, jetspec_counts


def merge_and_summarize_hybrid_drafts(
    suffix_batches: Iterable[Iterable[int]],
    jetspec_batches: Iterable[Iterable[int]],
    jetspec_raw_counts: Iterable[int],
    adapter_budget_counts: Iterable[int],
    *,
    num_speculative_tokens: int,
    suffix_budget_tokens: int,
    force_hybrid_proposals: bool,
) -> tuple[list[list[int]], list[int], list[int], dict[str, int | bool]]:
    suffixes = [
        [int(token_id) for token_id in suffix_tokens]
        for suffix_tokens in suffix_batches
    ]
    jetspecs = [
        [int(token_id) for token_id in jetspec_tokens]
        for jetspec_tokens in jetspec_batches
    ]
    jetspec_raw_count_list = [max(0, int(count)) for count in jetspec_raw_counts]
    adapter_budget_count_list = [
        max(0, int(count)) for count in adapter_budget_counts
    ]
    if _rust_merge_and_summarize_hybrid_drafts is not None:
        proposals, suffix_counts, jetspec_counts, values = (
            _rust_merge_and_summarize_hybrid_drafts(
                suffixes,
                jetspecs,
                jetspec_raw_count_list,
                adapter_budget_count_list,
                max(0, int(num_speculative_tokens)),
                max(0, int(suffix_budget_tokens)),
                bool(force_hybrid_proposals),
            )
        )
        summary: dict[str, int | bool] = {
            key: int(value)
            for key, value in zip(_HYBRID_COUNT_SUMMARY_KEYS, values)
        }
        summary["forced_suffix_and_jetspec_attempts"] = bool(
            summary["forced_suffix_and_jetspec_attempts"]
        )
        return (
            [[int(token_id) for token_id in proposal] for proposal in proposals],
            [int(count) for count in suffix_counts],
            [int(count) for count in jetspec_counts],
            summary,
        )

    proposals: list[list[int]] = []
    suffix_counts: list[int] = []
    jetspec_counts: list[int] = []
    budget = max(0, int(num_speculative_tokens))
    batch_count = max(len(suffixes), len(jetspecs))
    for index in range(batch_count):
        suffix_tokens = suffixes[index] if index < len(suffixes) else []
        jetspec_tokens = jetspecs[index] if index < len(jetspecs) else []
        adapter_budget = (
            adapter_budget_count_list[index]
            if index < len(adapter_budget_count_list)
            else 0
        )
        if force_hybrid_proposals and adapter_budget > 0 and jetspec_tokens:
            reserved_jetspec = min(budget, adapter_budget, len(jetspec_tokens))
            suffix_take = min(len(suffix_tokens), max(0, budget - reserved_jetspec))
            jetspec_take = min(len(jetspec_tokens), max(0, budget - suffix_take))
        else:
            suffix_take = min(len(suffix_tokens), budget)
            jetspec_take = min(len(jetspec_tokens), max(0, budget - suffix_take))
        proposals.append(suffix_tokens[:suffix_take] + jetspec_tokens[:jetspec_take])
        suffix_counts.append(suffix_take)
        jetspec_counts.append(jetspec_take)
    summary = summarize_hybrid_counts(
        suffix_counts,
        jetspec_counts,
        jetspec_raw_count_list,
        adapter_budget_count_list,
        num_speculative_tokens=num_speculative_tokens,
        suffix_budget_tokens=suffix_budget_tokens,
        force_hybrid_proposals=force_hybrid_proposals,
    )
    return proposals, suffix_counts, jetspec_counts, summary


def filter_jetspec_drafts(
    draft_batches: Iterable[Iterable[int]],
    prob_batches: Iterable[Iterable[float]] | None,
    budgets: Iterable[int],
    *,
    pad_token_id: int,
    min_token_prob: float,
) -> list[list[int]]:
    drafts = [[int(token_id) for token_id in row] for row in draft_batches]
    budget_list = [max(0, int(budget)) for budget in budgets]
    probs = (
        [[float(prob) for prob in row] for row in prob_batches]
        if prob_batches is not None
        else None
    )
    if _rust_filter_jetspec_drafts is not None:
        return [
            [int(token_id) for token_id in proposal]
            for proposal in _rust_filter_jetspec_drafts(
                drafts,
                probs or [],
                budget_list,
                int(pad_token_id),
                float(min_token_prob),
            )
        ]

    proposals: list[list[int]] = []
    for row_index, draft_row in enumerate(drafts):
        budget = budget_list[row_index] if row_index < len(budget_list) else 0
        prob_row = (
            probs[row_index]
            if probs is not None and row_index < len(probs)
            else None
        )
        proposal: list[int] = []
        for token_offset, token_id in enumerate(draft_row[:budget]):
            if token_id == pad_token_id:
                continue
            if (
                min_token_prob > 0
                and prob_row is not None
                and token_offset < len(prob_row)
                and prob_row[token_offset] < min_token_prob
            ):
                break
            proposal.append(token_id)
        proposals.append(proposal)
    return proposals


_HYBRID_COUNT_SUMMARY_KEYS = (
    "sequence_count",
    "proposed_tokens",
    "suffix_tokens",
    "jetspec_tokens",
    "jetspec_raw_tokens",
    "jetspec_unused_tokens",
    "adapter_budget_tokens",
    "suffix_budget_tokens",
    "suffix_attempted_sequences",
    "adapter_attempted_sequences",
    "adapter_empty_sequences",
    "sequences_filled_by_suffix",
    "sequences_with_suffix",
    "sequences_with_jetspec",
    "sequences_with_jetspec_raw",
    "sequences_without_proposal",
    "jetspec_filtered_tokens",
    "forced_suffix_and_jetspec_attempts",
)


def summarize_hybrid_counts(
    suffix_counts: Iterable[int],
    jetspec_counts: Iterable[int],
    jetspec_raw_counts: Iterable[int],
    adapter_budget_counts: Iterable[int],
    *,
    num_speculative_tokens: int,
    suffix_budget_tokens: int,
    force_hybrid_proposals: bool,
) -> dict[str, int | bool]:
    suffix_count_list = [max(0, int(count)) for count in suffix_counts]
    jetspec_count_list = [max(0, int(count)) for count in jetspec_counts]
    jetspec_raw_count_list = [max(0, int(count)) for count in jetspec_raw_counts]
    adapter_budget_count_list = [max(0, int(count)) for count in adapter_budget_counts]

    if _rust_summarize_hybrid_counts is not None:
        values = _rust_summarize_hybrid_counts(
            suffix_count_list,
            jetspec_count_list,
            jetspec_raw_count_list,
            adapter_budget_count_list,
            max(0, int(num_speculative_tokens)),
            max(0, int(suffix_budget_tokens)),
            bool(force_hybrid_proposals),
        )
        summary: dict[str, int | bool] = {
            key: int(value)
            for key, value in zip(_HYBRID_COUNT_SUMMARY_KEYS, values)
        }
        summary["forced_suffix_and_jetspec_attempts"] = bool(
            summary["forced_suffix_and_jetspec_attempts"]
        )
        return summary

    sequence_count = min(len(suffix_count_list), len(jetspec_count_list))
    proposed_counts = [
        suffix_count_list[index] + jetspec_count_list[index]
        for index in range(sequence_count)
    ]
    adapter_attempted_sequences = sum(
        1 for count in adapter_budget_count_list if count > 0
    )
    summary = {
        "sequence_count": sequence_count,
        "proposed_tokens": sum(proposed_counts),
        "suffix_tokens": sum(suffix_count_list[:sequence_count]),
        "jetspec_tokens": sum(jetspec_count_list[:sequence_count]),
        "jetspec_raw_tokens": sum(jetspec_raw_count_list),
        "adapter_budget_tokens": sum(adapter_budget_count_list),
        "suffix_budget_tokens": sequence_count * max(0, int(suffix_budget_tokens)),
        "suffix_attempted_sequences": (
            sequence_count if suffix_budget_tokens > 0 else 0
        ),
        "adapter_attempted_sequences": adapter_attempted_sequences,
        "adapter_empty_sequences": sum(
            1
            for budget, jetspec_count in zip(
                adapter_budget_count_list, jetspec_count_list
            )
            if budget > 0 and jetspec_count == 0
        ),
        "sequences_filled_by_suffix": sum(
            1
            for suffix_count in suffix_count_list[:sequence_count]
            if suffix_count >= num_speculative_tokens
        ),
        "sequences_with_suffix": sum(
            1 for count in suffix_count_list[:sequence_count] if count > 0
        ),
        "sequences_with_jetspec": sum(
            1 for count in jetspec_count_list[:sequence_count] if count > 0
        ),
        "sequences_with_jetspec_raw": sum(
            1 for count in jetspec_raw_count_list if count > 0
        ),
        "sequences_without_proposal": sum(1 for count in proposed_counts if count == 0),
        "jetspec_filtered_tokens": sum(
            max(0, int(budget) - int(jetspec_count))
            for budget, jetspec_count in zip(
                adapter_budget_count_list, jetspec_count_list
            )
        ),
    }
    summary["jetspec_unused_tokens"] = max(
        0, int(summary["jetspec_raw_tokens"]) - int(summary["jetspec_tokens"])
    )
    summary["forced_suffix_and_jetspec_attempts"] = bool(
        force_hybrid_proposals
        and suffix_budget_tokens > 0
        and sequence_count > 0
        and adapter_attempted_sequences == sequence_count
    )
    return summary
