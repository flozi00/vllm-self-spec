# Graph Report - vllm_plugin_live_train_spec  (2026-07-07)

## Corpus Check
- 24 files · ~43,473 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 666 nodes · 1757 edges · 56 communities (23 shown, 33 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 51 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Live LoRA Trainer|Live LoRA Trainer]]
- [[_COMMUNITY_JetSpec Adapter|JetSpec Adapter]]
- [[_COMMUNITY_Hybrid Proposer|Hybrid Proposer]]
- [[_COMMUNITY_App Request Handling|App Request Handling]]
- [[_COMMUNITY_Settings And Cache|Settings And Cache]]
- [[_COMMUNITY_Training Endpoints|Training Endpoints]]
- [[_COMMUNITY_Rust Suffix Backend|Rust Suffix Backend]]
- [[_COMMUNITY_Proxy Sample Extraction|Proxy Sample Extraction]]
- [[_COMMUNITY_Hub Checkpoints|Hub Checkpoints]]
- [[_COMMUNITY_Graphify Workflow|Graphify Workflow]]
- [[_COMMUNITY_Runtime Configuration|Runtime Configuration]]
- [[_COMMUNITY_Service Documentation|Service Documentation]]
- [[_COMMUNITY_Live Training Proxy|Live Training Proxy]]
- [[_COMMUNITY_Speculative Config|Speculative Config]]
- [[_COMMUNITY_Checkpoint Paths|Checkpoint Paths]]
- [[_COMMUNITY_Trainer Lifecycle|Trainer Lifecycle]]
- [[_COMMUNITY_Streaming Capture|Streaming Capture]]
- [[_COMMUNITY_Torch Defaults|Torch Defaults]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Crate Metadata|Crate Metadata]]
- [[_COMMUNITY_graphify reference extra exports and benchmark|graphify reference: extra exports and benchmark]]
- [[_COMMUNITY__env_first|_env_first]]
- [[_COMMUNITY_graphify reference query, path, explain|graphify reference: query, path, explain]]
- [[_COMMUNITY_graphify reference add a URL and watch a folder|graphify reference: add a URL and watch a folder]]
- [[_COMMUNITY_graphify reference commit hook and native CLAUDE.md integration|graphify reference: commit hook and native CLAUDE.md integration]]
- [[_COMMUNITY_graphify reference incremental update and cluster-only|graphify reference: incremental update and cluster-only]]
- [[_COMMUNITY_graphify reference GitHub clone and cross-repo merge|graphify reference: GitHub clone and cross-repo merge]]
- [[_COMMUNITY_graphify reference transcribe video and audio|graphify reference: transcribe video and audio]]
- [[_COMMUNITY_AGENTS|AGENTS.md]]
- [[_COMMUNITY_extraction-spec|extraction-spec.md]]
- [[_COMMUNITY_Exports Reference|Exports Reference]]
- [[_COMMUNITY_Extraction Subagent Prompt|Extraction Subagent Prompt]]
- [[_COMMUNITY_graphify merge-graphs|graphify merge-graphs]]
- [[_COMMUNITY_Post-Commit Hook|Post-Commit Hook]]
- [[_COMMUNITY_Constrained Query Expansion|Constrained Query Expansion]]
- [[_COMMUNITY_Query Flow|Query Flow]]
- [[_COMMUNITY_Video Audio Transcription|Video Audio Transcription]]
- [[_COMMUNITY_Incremental Update|Incremental Update]]
- [[_COMMUNITY_Build Cluster Outputs|Build Cluster Outputs]]
- [[_COMMUNITY_graphify Skill Documentation|graphify Skill Documentation]]
- [[_COMMUNITY_Existing Graph Fast Path|Existing Graph Fast Path]]
- [[_COMMUNITY_Graphify Full Pipeline|Graphify Full Pipeline]]
- [[_COMMUNITY_Graph Health Check|Graph Health Check]]
- [[_COMMUNITY_graph.json|graph.json]]
- [[_COMMUNITY_graphify|graphify]]
- [[_COMMUNITY_Semantic Extraction|Semantic Extraction]]
- [[_COMMUNITY_Structural Extraction|Structural Extraction]]
- [[_COMMUNITY_Dirty Graph Tolerance|Dirty Graph Tolerance]]
- [[_COMMUNITY_graphify Project Rules|graphify Project Rules]]
- [[_COMMUNITY_graphify Query First Rule|graphify Query First Rule]]
- [[_COMMUNITY_Post-Modify Update Rule|Post-Modify Update Rule]]
- [[_COMMUNITY_Wiki Navigation Rule|Wiki Navigation Rule]]
- [[_COMMUNITY_HybridSuffixJetSpecProposer|HybridSuffixJetSpecProposer]]
- [[_COMMUNITY_JetSpec Mode|JetSpec Mode]]
- [[_COMMUNITY_Plugin Hybrid Mode|Plugin Hybrid Mode]]
- [[_COMMUNITY_Suffix Decoding|Suffix Decoding]]

## God Nodes (most connected - your core abstractions)
1. `CpuSuffixCache` - 38 edges
2. `run_training()` - 37 edges
3. `run_training()` - 29 edges
4. `TrainingCoordinator` - 27 edges
5. `TrafficRecorder` - 20 edges
6. `VllmProcess` - 19 edges
7. `HybridSuffixJetSpecProposer` - 18 edges
8. `proxy()` - 17 edges
9. `SuffixCacheConfig` - 17 edges
10. `RustSuffixCache` - 16 edges

## Surprising Connections (you probably didn't know these)
- `Settings` --uses--> `CpuSuffixCache`  [INFERRED]
  app.py → suffix_cache.py
- `Settings` --uses--> `SuffixCacheConfig`  [INFERRED]
  app.py → suffix_cache.py
- `VllmProcess` --uses--> `DraftModelResolution`  [INFERRED]
  app.py → paths.py
- `VllmProcess` --uses--> `CpuSuffixCache`  [INFERRED]
  app.py → suffix_cache.py
- `VllmProcess` --uses--> `SuffixCacheConfig`  [INFERRED]
  app.py → suffix_cache.py

## Import Cycles
- None detected.

## Communities (56 total, 33 thin omitted)

### Community 0 - "Live LoRA Trainer"
Cohesion: 0.06
Nodes (102): _adapter_config_payload(), _attach_modelopt_nvfp4_sidecar_tensors(), _break_modelopt_nvfp4_bytes(), _build_dpo_example(), _build_sft_example(), _cached_modelopt_nvfp4_weight(), _causal_lm_hidden_states(), _collate_token_examples() (+94 more)

### Community 1 - "JetSpec Adapter"
Cohesion: 0.06
Nodes (97): JetSpecAdapter, JetSpecAdapterConfig, load_adapter(), load_checkpoint(), Any, PathLike, Small causal draft adapter used for JIT JetSpec experiments.      Native JetSpec, save_checkpoint() (+89 more)

### Community 2 - "Hybrid Proposer"
Cohesion: 0.07
Nodes (26): _checkpoint_from_latest_json(), _checkpoint_quality_allowed(), _env_bool_first(), _env_first(), _env_float(), _env_int(), _env_int_first(), _env_name_candidates() (+18 more)

### Community 3 - "App Request Handling"
Cohesion: 0.12
Nodes (40): _accepted_tokens_by_position(), _append_text_to_content(), _content_contains_text(), _content_text(), _default_enable_thinking(), _effective_accept_depth(), _env_json_obj(), _extract_completion() (+32 more)

### Community 4 - "Settings And Cache"
Cohesion: 0.11
Nodes (11): _custom_proposer_env_name(), _env_bool(), jit_rebuild_suffix_cache(), jit_sleep(), _messages_from_stored_prompt(), _suffix_cache_request_id(), TrafficRecorder, VllmProcess (+3 more)

### Community 5 - "Training Endpoints"
Cohesion: 0.15
Nodes (21): _active_live_lora_adapter_path(), _count_jsonl_rows(), _effective_training_snapshot(), health(), jit_sync_checkpoints(), jit_train_data(), jit_train_live_lora(), _latest_checkpoint_step() (+13 more)

### Community 6 - "Rust Suffix Backend"
Cohesion: 0.14
Nodes (14): Bound, HashMap, Option, PyModule, PyResult, filter_jetspec_drafts(), merge_and_summarize_hybrid_drafts(), merge_hybrid_drafts() (+6 more)

### Community 7 - "Proxy Sample Extraction"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 8 - "Hub Checkpoints"
Cohesion: 0.25
Nodes (21): _atomic_copy(), _atomic_write_json(), _checkpoint_name_from_latest(), _checkpoint_path_from_latest(), _checkpoint_step(), config_from_env(), download_newer_checkpoint(), _downloaded_latest_payload() (+13 more)

### Community 10 - "Runtime Configuration"
Cohesion: 0.18
Nodes (18): _bool_cli_option(), _env_bool_any(), _env_first_str(), _extra_arg_present(), _live_lora_serving_enabled(), _live_trainer_parallel_mode(), _mix_live_traffic_enabled(), _optional_int_any() (+10 more)

### Community 11 - "Service Documentation"
Cohesion: 0.11
Nodes (18): Draft Heads, Hub Checkpoint Sync, /jit Status Endpoints, Live SFT/DPO LoRA Training, Manager-Triggered JetSpec Training, Observability, Raw-Torch LoRA Trainer, Rust Suffix Backend (+10 more)

### Community 12 - "Live Training Proxy"
Cohesion: 0.17
Nodes (16): _append_live_training_rows(), _filtered_request_headers(), _filtered_response_headers(), _ingest_live_training_rows(), jit_train_dpo(), jit_train_sft(), _live_lora_proxy_enabled(), _model_wake_failed_response() (+8 more)

### Community 13 - "Speculative Config"
Cohesion: 0.15
Nodes (15): _auto_spec_method(), _custom_hybrid_speculative_config(), _default_ephemeral_data_dir(), _env_float(), _env_int(), _env_int_any(), _env_json_obj_any(), _load_speculative_config() (+7 more)

### Community 14 - "Checkpoint Paths"
Cohesion: 0.41
Nodes (12): checkpoint_dir_for_model(), checkpoint_dir_from_env(), _env_first_nonempty(), _env_nonempty(), latest_checkpoint_json_from_env(), _looks_like_hf_model_dir(), model_checkpoint_slug(), native_jetspec_draft_model_resolution_from_env() (+4 more)

### Community 15 - "Trainer Lifecycle"
Cohesion: 0.32
Nodes (3): _env_float_any(), jit_wake(), shutdown()

### Community 16 - "Streaming Capture"
Cohesion: 0.39
Nodes (3): _completion_payload(), _stream_response(), StreamingCompletionCapture

### Community 17 - "Torch Defaults"
Cohesion: 0.83
Nodes (3): _configure_torch_defaults(), _running_pip(), _torch_float32_matmul_precision()

### Community 20 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 21 - "_env_first"
Cohesion: 0.25
Nodes (7): _configure_logging(), _env_first(), _force_thinking_marker(), _force_thinking_system_prompt(), _is_ephemeral_runtime_path(), _runtime_data_path(), startup()

### Community 22 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 23 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 24 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 25 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

## Ambiguous Edges - Review These
- `Speculative Decoding` → `arctic-inference`  [AMBIGUOUS]
  requirements.txt · relation: conceptually_related_to

## Knowledge Gaps
- **76 isolated node(s):** `LoRALinear`, `vllm-dflash-jit-rust`, `Usage`, `What graphify is for`, `Step 0 - GitHub repos and multi-path merge (only if a URL or several paths)` (+71 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **33 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Speculative Decoding` and `arctic-inference`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `CpuSuffixCache` connect `Hybrid Proposer` to `App Request Handling`, `Settings And Cache`, `Training Endpoints`, `Speculative Config`, `Streaming Capture`?**
  _High betweenness centrality (0.139) - this node is a cross-community bridge._
- **Why does `run_training()` connect `Live LoRA Trainer` to `Settings And Cache`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Why does `SuffixCacheConfig` connect `Hybrid Proposer` to `App Request Handling`, `Settings And Cache`, `Training Endpoints`, `Speculative Config`, `Streaming Capture`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `CpuSuffixCache` (e.g. with `Settings` and `StreamingCompletionCapture`) actually correct?**
  _`CpuSuffixCache` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `JIT JetSpec speculation helpers for a vLLM development service.`, `Experimental vLLM custom proposer that combines CPU suffix and JetSpec drafts.`, `Small causal draft adapter used for JIT JetSpec experiments.      Native JetSpec` to the rest of the system?**
  _85 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Live LoRA Trainer` be split into smaller, more focused modules?**
  _Cohesion score 0.05836713101745724 - nodes in this community are weakly interconnected._