from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable

from .jetspec_adapter import load_adapter
from .paths import latest_checkpoint_json_from_env
from .suffix_cache import (
    CpuSuffixCache,
    SuffixCacheConfig,
    filter_jetspec_drafts,
    merge_and_summarize_hybrid_drafts,
    summarize_hybrid_counts,
)


logger = logging.getLogger("vllm_dflash_jit.hybrid_proposer")
PLUGIN_ENV_PREFIX_ALIASES = (
    ("VLLM_JETSPEC_", "JETSPEC_PLUGIN_"),
    ("VLLM_SUFFIX_", "JETSPEC_SUFFIX_"),
)


def _env_name_candidates(name: str) -> tuple[str, ...]:
    for prefix, alias_prefix in PLUGIN_ENV_PREFIX_ALIASES:
        if name.startswith(prefix):
            return (name, f"{alias_prefix}{name[len(prefix):]}")
    return (name,)


def _env_int(name: str, default: int) -> int:
    raw = _env_first(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_int_first(*names: str, default: int) -> int:
    raw = _env_first(*names)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _env_first(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        for candidate in _env_name_candidates(name):
            raw = os.getenv(candidate)
            if raw is not None and raw != "":
                return raw
    return default


def _env_bool_first(*names: str, default: bool) -> bool:
    raw = _env_first(*names)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class JetSpecDraftRuntime:
    def __init__(self, checkpoint_path: str | None, *, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self._loaded_path: Path | None = None
        self._loaded_mtime: float | None = None
        self.reload_seconds = _env_float(
            "VLLM_JETSPEC_ADAPTER_RELOAD_SECONDS",
            5.0,
        )
        self.min_token_prob = _env_float(
            "VLLM_JETSPEC_MIN_TOKEN_PROB",
            0.30,
        )
        self._last_load_check = 0.0

    def propose(self, context: list[int], max_tokens: int) -> list[int]:
        return self.propose_many([context], [max_tokens])[0]

    def propose_many(
        self, contexts: list[list[int]], max_tokens_per_context: list[int]
    ) -> list[list[int]]:
        budgets = [max(0, int(budget)) for budget in max_tokens_per_context]
        if not contexts:
            return []
        if not any(budget > 0 for budget in budgets):
            return [[] for _context in contexts]
        if len(budgets) < len(contexts):
            budgets.extend([0] * (len(contexts) - len(budgets)))
        active = [
            (index, context, budgets[index])
            for index, context in enumerate(contexts)
            if budgets[index] > 0 and context
        ]
        proposals: list[list[int]] = [[] for _context in contexts]
        if not active:
            return proposals
        model = self._load_if_needed()
        if model is None:
            return proposals

        import torch

        config = model.adapter_config
        max_budget = min(
            max(budget for _index, _context, budget in active),
            config.num_speculative_tokens,
        )
        trimmed_contexts = [context[-256:] for _index, context, _budget in active]
        max_len = max(len(context) for context in trimmed_contexts)
        rows = [
            [config.pad_token_id] * (max_len - len(context)) + context
            for context in trimmed_contexts
        ]
        tensor = torch.tensor(rows, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = model(tensor)
            draft_logits = logits[:, :max_budget, :]
            token_scores, token_ids = draft_logits.max(dim=-1)
            drafts = token_ids.tolist()
            if self.min_token_prob > 0:
                token_probs = (token_scores - draft_logits.logsumexp(dim=-1)).exp()
                draft_probs = token_probs.tolist()
            else:
                draft_probs = None
        active_proposals = filter_jetspec_drafts(
            drafts,
            draft_probs,
            [budget for _index, _context, budget in active],
            pad_token_id=config.pad_token_id,
            min_token_prob=self.min_token_prob,
        )
        for active_index, (original_index, _context, budget) in enumerate(active):
            proposal = active_proposals[active_index] if active_index < len(active_proposals) else []
            proposals[original_index] = proposal[:budget]
        return proposals

    def _load_if_needed(self):
        now = time.monotonic()
        if (
            self.reload_seconds > 0
            and now - self._last_load_check < self.reload_seconds
        ):
            return self.model
        self._last_load_check = now
        path = self._resolve_checkpoint_path()
        if path is None or not path.exists():
            return self.model
        mtime = path.stat().st_mtime
        if (
            self.model is not None
            and self._loaded_path == path
            and self._loaded_mtime == mtime
        ):
            return self.model
        try:
            self.model, _payload = load_adapter(path, map_location=self.device)
            self._loaded_path = path
            self._loaded_mtime = mtime
            logger.info("Loaded JetSpec draft adapter checkpoint %s", path)
        except Exception:
            logger.exception("Failed to load JetSpec draft adapter checkpoint %s", path)
            self.model = None
            self._loaded_path = None
            self._loaded_mtime = None
        return self.model

    def _resolve_checkpoint_path(self) -> Path | None:
        if self.checkpoint_path:
            path = Path(self.checkpoint_path)
            if path.is_dir():
                return _latest_checkpoint_in_dir(path)
            if path.suffix == ".json":
                if path.exists():
                    return _checkpoint_from_latest_json(path)
                return _latest_checkpoint_in_dir(path.parent)
            return path
        latest_path = Path(
            latest_checkpoint_json_from_env(
                _env_first(
                    "VLLM_JETSPEC_MODEL", default="Qwen/Qwen3-8B"
                ),
                _env_first(
                    "VLLM_JETSPEC_DATA_DIR",
                    default="/data/vllm_jetspec",
                ),
            )
        )
        if latest_path.exists():
            return _checkpoint_from_latest_json(latest_path)
        return _latest_checkpoint_in_dir(latest_path.parent)


class ProposerStats:
    def __init__(self, path: str | None, *, flush_seconds: float, flush_every: int):
        self.path = Path(path) if path else None
        self.flush_seconds = flush_seconds
        self.flush_every = flush_every
        self._lock = threading.Lock()
        self._last_flush = 0.0
        now = time.time()
        self._stats: dict[str, Any] = {
            "version": 1,
            "started_at": now,
            "updated_at": now,
            "calls": 0,
            "sequences": 0,
            "requested_tokens": 0,
            "proposed_tokens": 0,
            "suffix_tokens": 0,
            "jetspec_tokens": 0,
            "sequences_with_suffix": 0,
            "sequences_with_jetspec": 0,
            "sequences_without_proposal": 0,
        }

    def record(
        self,
        *,
        num_speculative_tokens: int,
        configured_num_speculative_tokens: int | None = None,
        min_adapter_tokens: int,
        max_adapter_tokens: int,
        suffix_counts: list[int],
        jetspec_counts: list[int],
        jetspec: JetSpecDraftRuntime,
        adapter_budget_counts: list[int],
        jetspec_raw_counts: list[int] | None = None,
        suffix_budget_tokens: int,
        suffix_cache: CpuSuffixCache,
        force_hybrid_proposals: bool,
        count_summary: dict[str, int | bool] | None = None,
        phase_timings_ms: dict[str, float] | None = None,
        context_window_tokens: int | None = None,
    ) -> None:
        record_started = time.perf_counter()
        now = time.time()
        if jetspec_raw_counts is None:
            jetspec_raw_counts = list(jetspec_counts)
        if count_summary is None:
            count_summary = summarize_hybrid_counts(
                suffix_counts,
                jetspec_counts,
                jetspec_raw_counts,
                adapter_budget_counts,
                num_speculative_tokens=num_speculative_tokens,
                suffix_budget_tokens=suffix_budget_tokens,
                force_hybrid_proposals=force_hybrid_proposals,
            )
        sequence_count = int(count_summary["sequence_count"])
        with self._lock:
            self._stats["calls"] += 1
            self._stats["sequences"] += sequence_count
            self._stats["requested_tokens"] += (
                sequence_count * num_speculative_tokens
            )
            self._stats["proposed_tokens"] += int(count_summary["proposed_tokens"])
            self._stats["suffix_tokens"] += int(count_summary["suffix_tokens"])
            self._stats["jetspec_tokens"] += int(count_summary["jetspec_tokens"])
            self._stats["jetspec_raw_tokens"] = (
                self._stats.get("jetspec_raw_tokens", 0)
                + int(count_summary["jetspec_raw_tokens"])
            )
            self._stats["jetspec_unused_tokens"] = (
                self._stats.get("jetspec_unused_tokens", 0)
                + int(count_summary["jetspec_unused_tokens"])
            )
            self._stats["adapter_budget_tokens"] = (
                self._stats.get("adapter_budget_tokens", 0)
                + int(count_summary["adapter_budget_tokens"])
            )
            self._stats["suffix_budget_tokens"] = (
                self._stats.get("suffix_budget_tokens", 0)
                + int(count_summary["suffix_budget_tokens"])
            )
            self._stats["suffix_attempted_sequences"] = (
                self._stats.get("suffix_attempted_sequences", 0)
                + int(count_summary["suffix_attempted_sequences"])
            )
            self._stats["adapter_attempted_sequences"] = (
                self._stats.get("adapter_attempted_sequences", 0)
                + int(count_summary["adapter_attempted_sequences"])
            )
            self._stats["adapter_empty_sequences"] = (
                self._stats.get("adapter_empty_sequences", 0)
                + int(count_summary["adapter_empty_sequences"])
            )
            self._stats["sequences_filled_by_suffix"] = (
                self._stats.get("sequences_filled_by_suffix", 0)
                + int(count_summary["sequences_filled_by_suffix"])
            )
            self._stats["sequences_with_suffix"] += int(
                count_summary["sequences_with_suffix"]
            )
            self._stats["sequences_with_jetspec"] += int(
                count_summary["sequences_with_jetspec"]
            )
            self._stats["sequences_with_jetspec_raw"] = (
                self._stats.get("sequences_with_jetspec_raw", 0)
                + int(count_summary["sequences_with_jetspec_raw"])
            )
            self._stats["sequences_without_proposal"] += int(
                count_summary["sequences_without_proposal"]
            )
            self._stats["updated_at"] = now
            self._stats["num_speculative_tokens"] = num_speculative_tokens
            self._stats["configured_num_speculative_tokens"] = (
                configured_num_speculative_tokens
                if configured_num_speculative_tokens is not None
                else num_speculative_tokens
            )
            self._stats["min_adapter_tokens"] = min_adapter_tokens
            self._stats["max_adapter_tokens"] = max_adapter_tokens
            if context_window_tokens is not None:
                self._stats["context_window_tokens"] = int(context_window_tokens)
            self._stats["suffix_budget_per_sequence"] = suffix_budget_tokens
            loaded_path = getattr(jetspec, "_loaded_path", None)
            self._stats["jetspec_checkpoint_path"] = (
                str(loaded_path) if loaded_path else ""
            )
            self._stats["jetspec_checkpoint_mtime"] = getattr(
                jetspec, "_loaded_mtime", None
            )
            self._stats["jetspec_adapter_loaded"] = (
                getattr(jetspec, "model", None) is not None
            )
            self._stats["jetspec_adapter_reload_seconds"] = getattr(
                jetspec, "reload_seconds", None
            )
            self._stats["jetspec_min_token_prob"] = getattr(
                jetspec, "min_token_prob", None
            )
            self._stats["jetspec_filtered_tokens"] = (
                self._stats.get("jetspec_filtered_tokens", 0)
                + int(count_summary["jetspec_filtered_tokens"])
            )
            self._stats["hybrid_mode"] = "suffix_then_jetspec"
            self._stats["force_hybrid_proposals"] = force_hybrid_proposals
            self._stats["last_call_suffix_attempted_sequences"] = (
                int(count_summary["suffix_attempted_sequences"])
            )
            self._stats["last_call_adapter_attempted_sequences"] = (
                int(count_summary["adapter_attempted_sequences"])
            )
            self._stats["forced_suffix_and_jetspec_attempts"] = bool(
                count_summary["forced_suffix_and_jetspec_attempts"]
            )
            self._stats["suffix_cache_backend"] = getattr(
                suffix_cache, "backend_name", "python"
            )
            self._stats["suffix_cache_requests"] = suffix_cache.request_count
            self._stats["suffix_cache_suffixes"] = suffix_cache.suffix_count
            self._stats["suffix_min_match_tokens"] = (
                suffix_cache.config.min_match_tokens
            )
            self._stats["suffix_min_token_prob"] = suffix_cache.config.min_token_prob
            self._stats["suffix_max_spec_factor"] = suffix_cache.config.max_spec_factor
            self._stats["suffix_max_tree_depth"] = suffix_cache.config.max_tree_depth
            self._stats["in_context_suffix_enabled"] = _env_bool_first(
                "VLLM_JETSPEC_IN_CONTEXT_SUFFIX", default=True
            )
            self._stats["in_context_suffix_bootstrap"] = _env_bool_first(
                "VLLM_JETSPEC_IN_CONTEXT_SUFFIX_BOOTSTRAP", default=True
            )
            self._stats["in_context_suffix_window"] = _env_int(
                "VLLM_JETSPEC_IN_CONTEXT_SUFFIX_WINDOW", 8192
            )
            self._stats["in_context_suffix_min_match_tokens"] = _env_int(
                "VLLM_JETSPEC_IN_CONTEXT_SUFFIX_MIN_MATCH",
                suffix_cache.config.min_match_tokens,
            )
            self._stats["proposal_fill_rate"] = _safe_ratio(
                self._stats["proposed_tokens"], self._stats["requested_tokens"]
            )
            self._stats["average_requested_tokens_per_sequence"] = _safe_ratio(
                self._stats["requested_tokens"], self._stats["sequences"]
            )
            self._stats["average_proposed_tokens_per_sequence"] = _safe_ratio(
                self._stats["proposed_tokens"], self._stats["sequences"]
            )
            self._stats["average_suffix_tokens_per_sequence"] = _safe_ratio(
                self._stats["suffix_tokens"], self._stats["sequences"]
            )
            self._stats["average_jetspec_tokens_per_sequence"] = _safe_ratio(
                self._stats["jetspec_tokens"], self._stats["sequences"]
            )
            self._stats["average_jetspec_raw_tokens_per_sequence"] = _safe_ratio(
                self._stats.get("jetspec_raw_tokens", 0),
                self._stats["sequences"],
            )
            self._stats["average_adapter_budget_tokens_per_sequence"] = _safe_ratio(
                self._stats.get("adapter_budget_tokens", 0),
                self._stats["sequences"],
            )
            self._stats["suffix_share_of_proposed_tokens"] = _safe_ratio(
                self._stats["suffix_tokens"], self._stats["proposed_tokens"]
            )
            self._stats["jetspec_share_of_proposed_tokens"] = _safe_ratio(
                self._stats["jetspec_tokens"], self._stats["proposed_tokens"]
            )
            self._stats["jetspec_raw_share_of_attempted_tokens"] = _safe_ratio(
                self._stats.get("jetspec_raw_tokens", 0),
                self._stats.get("adapter_budget_tokens", 0),
            )
            self._stats["dominant_draft_source"] = (
                "jetspec"
                if self._stats["jetspec_share_of_proposed_tokens"]
                > self._stats["suffix_share_of_proposed_tokens"]
                else "suffix"
            )
            if phase_timings_ms is not None:
                phase_timings = dict(phase_timings_ms)
                phase_timings["record_stats"] = (
                    time.perf_counter() - record_started
                ) * 1000.0
                total_ms = float(sum(phase_timings.values()))
                self._stats["last_call_phase_ms"] = phase_timings
                self._stats["last_call_total_ms"] = total_ms
                self._stats["total_propose_ms"] = (
                    float(self._stats.get("total_propose_ms", 0.0)) + total_ms
                )
                self._stats["average_propose_ms"] = _safe_ratio(
                    self._stats["total_propose_ms"], self._stats["calls"]
                )
                cumulative = dict(
                    self._stats.get("cumulative_phase_ms", {})
                    if isinstance(self._stats.get("cumulative_phase_ms"), dict)
                    else {}
                )
                average: dict[str, float] = {}
                for phase, elapsed_ms in phase_timings.items():
                    cumulative[phase] = float(cumulative.get(phase, 0.0)) + float(
                        elapsed_ms
                    )
                    average[phase] = _safe_ratio(cumulative[phase], self._stats["calls"])
                self._stats["cumulative_phase_ms"] = cumulative
                self._stats["average_phase_ms"] = average
            should_flush = self.path is not None and (
                self._last_flush == 0.0
                or now - self._last_flush >= self.flush_seconds
                or (
                    self.flush_every > 0
                    and self._stats["calls"] % self.flush_every == 0
                )
            )
            if should_flush:
                self._flush_locked(now)

    def _flush_locked(self, now: float) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_name(
                f"{self.path.name}.{os.getpid()}.{id(self):x}.tmp"
            )
            tmp_path.write_text(
                json.dumps(self._stats, indent=2) + "\n", encoding="utf-8"
            )
            tmp_path.replace(self.path)
            self._last_flush = now
        except OSError:
            logger.exception("Failed to write JetSpec proposer stats to %s", self.path)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


class HybridSuffixJetSpecProposer:
    """Experimental vLLM custom proposer that combines CPU suffix and JetSpec drafts.

    Configure with:

    --speculative-config '{"method":"custom_class",
      "model":"vllm_dflash_jit.hybrid_proposer.HybridSuffixJetSpecProposer",
      "num_speculative_tokens":8}'

    The custom proposer API is intentionally narrow. This implementation uses
    duck-typed InputBatch access so it can track nightly vLLM API movement while
    remaining importable in tests.
    """

    def __init__(self, vllm_config: Any):
        spec_config = getattr(vllm_config, "speculative_config", None)
        self.num_speculative_tokens = int(
            getattr(spec_config, "num_speculative_tokens", None)
            or _env_int(
                "VLLM_JETSPEC_NUM_SPECULATIVE_TOKENS",
                8,
            )
        )
        configured_num_speculative_tokens = self.num_speculative_tokens
        max_proposed_tokens = int(
            _env_first(
                "VLLM_JETSPEC_MAX_PROPOSED_TOKENS",
                default="0",
            )
            or "0"
        )
        if max_proposed_tokens > 0:
            self.num_speculative_tokens = max(
                1, min(self.num_speculative_tokens, max_proposed_tokens)
            )
        self.max_num_speculative_tokens = self.num_speculative_tokens
        self.force_hybrid_proposals = _env_bool_first(
            "VLLM_JETSPEC_FORCE_HYBRID_PROPOSALS",
            default=True,
        )
        self.max_adapter_tokens = int(
            _env_first(
                "VLLM_JETSPEC_MAX_ADAPTER_TOKENS",
                default=str(self.num_speculative_tokens),
            )
            or str(self.num_speculative_tokens)
        )
        self.max_adapter_tokens = max(
            0, min(self.num_speculative_tokens, self.max_adapter_tokens)
        )
        if self.force_hybrid_proposals and self.num_speculative_tokens > 0:
            self.max_adapter_tokens = max(1, self.max_adapter_tokens)
        self.min_adapter_tokens = int(
            _env_first(
                "VLLM_JETSPEC_MIN_ADAPTER_TOKENS",
                default="1",
            )
            or "1"
        )
        self.min_adapter_tokens = max(
            0,
            min(
                self.num_speculative_tokens,
                self.max_adapter_tokens,
                self.min_adapter_tokens,
            ),
        )
        if self.force_hybrid_proposals and self.max_adapter_tokens > 0:
            self.min_adapter_tokens = max(1, self.min_adapter_tokens)
        suffix_config = SuffixCacheConfig(
            max_tree_depth=_env_int("VLLM_SUFFIX_MAX_TREE_DEPTH", 24),
            max_cached_requests=_env_int("VLLM_SUFFIX_MAX_CACHED_REQUESTS", 10_000),
            max_spec_factor=_env_float("VLLM_SUFFIX_MAX_SPEC_FACTOR", 1.0),
            min_token_prob=_env_float("VLLM_SUFFIX_MIN_TOKEN_PROB", 0.1),
            min_match_tokens=_env_int("VLLM_SUFFIX_MIN_MATCH_TOKENS", 1),
        )
        self.context_window_tokens = max(
            1,
            _env_int(
                "VLLM_JETSPEC_CONTEXT_WINDOW_TOKENS",
                max(256, suffix_config.max_tree_depth),
            ),
        )
        suffix_cache_path = _env_first("VLLM_JETSPEC_SUFFIX_CACHE_PATH")
        self.suffix_cache_path = Path(suffix_cache_path) if suffix_cache_path else None
        self.suffix_config = suffix_config
        self.suffix_cache = (
            CpuSuffixCache.load(self.suffix_cache_path, suffix_config)
            if self.suffix_cache_path
            else CpuSuffixCache(suffix_config)
        )
        self._suffix_cache_mtime_ns = self._read_suffix_cache_mtime_ns()
        self._suffix_reload_seconds = _env_float(
            "VLLM_JETSPEC_SUFFIX_RELOAD_SECONDS",
            2.0,
        )
        self._last_suffix_reload_check = 0.0
        self.jetspec = JetSpecDraftRuntime(
            _env_first("VLLM_JETSPEC_ADAPTER_CHECKPOINT"),
            device=_env_first(
                "VLLM_JETSPEC_PROPOSER_DEVICE",
                default="cuda",
            )
            or "cuda",
        )
        self.stats = ProposerStats(
            _env_first(
                "VLLM_JETSPEC_PROPOSER_STATS_PATH",
                default=str(
                    Path(
                        _env_first(
                            "VLLM_JETSPEC_DATA_DIR",
                            default="/data/vllm_jetspec",
                        )
                        or "/data/vllm_jetspec"
                    )
                    / "proposer_stats.json"
                ),
            ),
            flush_seconds=_env_float(
                "VLLM_JETSPEC_PROPOSER_STATS_FLUSH_SECONDS",
                5.0,
            ),
            flush_every=_env_int(
                "VLLM_JETSPEC_PROPOSER_STATS_FLUSH_EVERY",
                32,
            ),
        )
        logger.info(
            "Hybrid suffix+JetSpec proposer initialized num_speculative_tokens=%d configured_num_speculative_tokens=%d min_adapter_tokens=%d max_adapter_tokens=%d force_hybrid_proposals=%s suffix_cache_backend=%s suffix_cache_requests=%d suffix_cache_suffixes=%d",
            self.num_speculative_tokens,
            configured_num_speculative_tokens,
            self.min_adapter_tokens,
            self.max_adapter_tokens,
            self.force_hybrid_proposals,
            getattr(self.suffix_cache, "backend_name", "python"),
            self.suffix_cache.request_count,
            self.suffix_cache.suffix_count,
        )

    def propose(
        self,
        *args: Any,
        slot_mappings: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> list[list[int]]:
        phase_timings_ms: dict[str, float] = {}

        def mark(phase: str, started: float) -> None:
            phase_timings_ms[phase] = phase_timings_ms.get(phase, 0.0) + (
                time.perf_counter() - started
            ) * 1000.0

        proposals: list[list[int]] = []
        suffix_counts: list[int] = []
        jetspec_counts: list[int] = []
        adapter_budget_counts: list[int] = []
        phase_started = time.perf_counter()
        self._maybe_reload_suffix_cache()
        mark("reload_suffix_cache", phase_started)
        phase_started = time.perf_counter()
        contexts = self._contexts_from_args(*args)
        mark("extract_contexts", phase_started)
        phase_started = time.perf_counter()
        proposal_token_budget = self._proposal_token_budget()
        suffix_budget = proposal_token_budget
        max_adapter_tokens = min(self.max_adapter_tokens, proposal_token_budget)
        if (
            self.force_hybrid_proposals
            and proposal_token_budget > 0
            and self.max_adapter_tokens > 0
        ):
            max_adapter_tokens = max(1, max_adapter_tokens)
        min_adapter_tokens = min(self.min_adapter_tokens, max_adapter_tokens)
        mark("configure_budget", phase_started)
        phase_started = time.perf_counter()
        (
            suffix_batches,
            adapter_contexts,
            adapter_budget_counts,
        ) = self.suffix_cache.propose_many_with_adapter_inputs(
            contexts,
            suffix_max_tokens=suffix_budget,
            max_adapter_tokens=max_adapter_tokens,
        )
        mark("suffix_lookup", phase_started)
        phase_started = time.perf_counter()
        jetspec_batches = self._jetspec_propose_many(
            adapter_contexts, adapter_budget_counts
        )
        mark("jetspec_draft", phase_started)
        phase_started = time.perf_counter()
        jetspec_raw_counts = [len(batch) for batch in jetspec_batches]
        (
            proposals,
            suffix_counts,
            jetspec_counts,
            count_summary,
        ) = merge_and_summarize_hybrid_drafts(
            suffix_batches,
            jetspec_batches,
            jetspec_raw_counts,
            adapter_budget_counts,
            num_speculative_tokens=proposal_token_budget,
            suffix_budget_tokens=suffix_budget,
            force_hybrid_proposals=self.force_hybrid_proposals,
        )
        mark("merge_and_summarize", phase_started)
        self.stats.record(
            num_speculative_tokens=proposal_token_budget,
            configured_num_speculative_tokens=self.max_num_speculative_tokens,
            min_adapter_tokens=min_adapter_tokens,
            max_adapter_tokens=max_adapter_tokens,
            suffix_counts=suffix_counts,
            jetspec_counts=jetspec_counts,
            jetspec=self.jetspec,
            adapter_budget_counts=adapter_budget_counts,
            jetspec_raw_counts=jetspec_raw_counts,
            suffix_budget_tokens=suffix_budget,
            suffix_cache=self.suffix_cache,
            force_hybrid_proposals=self.force_hybrid_proposals,
            count_summary=count_summary,
            phase_timings_ms=phase_timings_ms,
            context_window_tokens=self.context_window_tokens,
        )
        return proposals

    def _proposal_token_budget(self) -> int:
        return self.max_num_speculative_tokens

    def _jetspec_propose_many(
        self, contexts: list[list[int]], budgets: list[int]
    ) -> list[list[int]]:
        propose_many = getattr(self.jetspec, "propose_many", None)
        if propose_many is not None:
            return propose_many(contexts, budgets)
        return [
            self.jetspec.propose(context, budget) if budget > 0 else []
            for context, budget in zip(contexts, budgets)
        ]

    def _read_suffix_cache_mtime_ns(self) -> int:
        if self.suffix_cache_path is None:
            return 0
        try:
            return self.suffix_cache_path.stat().st_mtime_ns
        except OSError:
            return 0

    def _maybe_reload_suffix_cache(self) -> None:
        if self.suffix_cache_path is None or self._suffix_reload_seconds < 0:
            return
        now = time.monotonic()
        if (
            self._suffix_reload_seconds > 0
            and now - self._last_suffix_reload_check < self._suffix_reload_seconds
        ):
            return
        self._last_suffix_reload_check = now
        mtime_ns = self._read_suffix_cache_mtime_ns()
        if mtime_ns == self._suffix_cache_mtime_ns:
            return
        try:
            self.suffix_cache = CpuSuffixCache.load(
                self.suffix_cache_path, self.suffix_config
            )
            self._suffix_cache_mtime_ns = mtime_ns
            logger.info(
                "Reloaded suffix cache path=%s requests=%d suffixes=%d",
                self.suffix_cache_path,
                self.suffix_cache.request_count,
                self.suffix_cache.suffix_count,
            )
        except Exception:
            logger.exception("Failed to reload suffix cache %s", self.suffix_cache_path)

    def _contexts_from_args(self, *args: Any) -> list[list[int]]:
        if len(args) >= 3:
            sampled_token_ids = args[0]
            num_tokens_no_spec = self._coerce_lengths(args[1])
            token_rows = args[2]
        elif len(args) >= 2:
            input_batch = args[0]
            sampled_token_ids = args[1]
            token_rows = self._token_rows_source(input_batch)
            num_tokens_no_spec = self._coerce_lengths(
                getattr(input_batch, "num_tokens_no_spec", [])
            )
        else:
            return []

        batch_size = len(sampled_token_ids)
        contexts: list[list[int]] = []
        for index in range(batch_size):
            sampled = sampled_token_ids[index] if index < len(sampled_token_ids) else []
            if not self._has_sampled_token(sampled):
                contexts.append([])
                continue
            length = (
                num_tokens_no_spec[index] if index < len(num_tokens_no_spec) else None
            )
            row = self._row_at(token_rows, index)
            contexts.append(
                self._coerce_context_row(
                    row,
                    length=length,
                    max_tokens=self.context_window_tokens,
                )
            )
        return contexts

    def _has_sampled_token(self, sampled: Any) -> bool:
        if sampled is None:
            return False
        if hasattr(sampled, "tolist"):
            sampled = sampled.tolist()
        if isinstance(sampled, (bytes, str)):
            return bool(sampled)
        if isinstance(sampled, Iterable):
            return any(True for _item in sampled)
        return True

    def _token_rows_source(self, input_batch: Any) -> Any:
        for attr in ("token_ids_cpu", "input_ids_cpu", "token_ids", "input_ids"):
            rows = getattr(input_batch, attr, None)
            if rows is not None:
                return rows
        return None

    def _row_at(self, rows: Any, index: int) -> Any:
        if rows is None:
            return None
        if isinstance(rows, dict):
            rows = list(rows.values())
        if isinstance(rows, (bytes, str)):
            return None
        try:
            return rows[index]
        except (IndexError, KeyError, TypeError):
            return None

    def _coerce_context_row(
        self,
        row: Any,
        *,
        length: int | None,
        max_tokens: int,
    ) -> list[int]:
        if row is None or isinstance(row, (bytes, str)):
            return []
        stop = max(0, int(length)) if length is not None else None
        try:
            row_length = len(row)
        except TypeError:
            row_length = None
        if stop is not None and row_length is not None:
            stop = min(stop, row_length)
        limit = max(1, int(max_tokens))
        if stop is None:
            start = -limit
        else:
            start = max(0, stop - limit)
        try:
            if stop is None:
                row = row[start:]
            else:
                row = row[start:stop]
        except TypeError:
            pass
        if hasattr(row, "detach"):
            row = row.detach()
        if hasattr(row, "cpu"):
            row = row.cpu()
        if hasattr(row, "tolist"):
            row = row.tolist()
        if isinstance(row, dict):
            row = list(row.values())
        if not isinstance(row, Iterable) or isinstance(row, (bytes, str)):
            return []
        try:
            return [int(token_id) for token_id in row]
        except TypeError:
            return []

    def _token_rows(self, input_batch: Any) -> list[list[int]]:
        rows = self._token_rows_source(input_batch)
        parsed = self._coerce_rows(rows)
        if parsed:
            return parsed
        return []

    def _coerce_lengths(self, values: Any) -> list[int]:
        if values is None:
            return []
        if hasattr(values, "tolist"):
            values = values.tolist()
        if isinstance(values, dict):
            values = list(values.values())
        if isinstance(values, (bytes, str)):
            return []
        try:
            return [int(value) for value in values]
        except TypeError:
            return []

    def _coerce_rows(self, rows: Any) -> list[list[int]]:
        if rows is None:
            return []
        if hasattr(rows, "tolist"):
            rows = rows.tolist()
        if isinstance(rows, dict):
            rows = list(rows.values())
        if not isinstance(rows, Iterable) or isinstance(rows, (bytes, str)):
            return []
        parsed: list[list[int]] = []
        for row in rows:
            if hasattr(row, "tolist"):
                row = row.tolist()
            if isinstance(row, (bytes, str)):
                continue
            try:
                parsed.append([int(token_id) for token_id in row])
            except TypeError:
                continue
        return parsed


def _checkpoint_from_latest_json(latest_path: Path) -> Path | None:
    if not latest_path.exists():
        return None
    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not _checkpoint_quality_allowed(payload, latest_path):
        return None
    checkpoint = payload.get("checkpoint") or payload.get("checkpoint_path")
    if not checkpoint:
        return None
    path = Path(str(checkpoint))
    if not path.is_absolute():
        path = latest_path.parent / path
    if path.exists():
        return path
    return _latest_checkpoint_in_dir(latest_path.parent)


def _checkpoint_quality_allowed(payload: dict[str, Any], latest_path: Path) -> bool:
    min_top1 = _env_float("VLLM_JETSPEC_MIN_CHECKPOINT_TOP1_ACCURACY", 0.0)
    if min_top1 <= 0:
        return True
    quality = payload.get("quality")
    if not isinstance(quality, dict) or "top1_accuracy" not in quality:
        return True
    try:
        top1_accuracy = float(quality.get("top1_accuracy") or 0.0)
    except (TypeError, ValueError):
        top1_accuracy = 0.0
    if top1_accuracy >= min_top1:
        return True
    logger.warning(
        "Skipping JetSpec checkpoint from %s because top1_accuracy %.4f is below %.4f",
        latest_path,
        top1_accuracy,
        min_top1,
    )
    return False


def _latest_checkpoint_in_dir(checkpoint_dir: Path) -> Path | None:
    try:
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-step-*.pt"))
    except OSError:
        return None
    return checkpoints[-1] if checkpoints else None
