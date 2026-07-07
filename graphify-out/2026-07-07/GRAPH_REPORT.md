# Graph Report - .  (2026-07-07)

## Corpus Check
- Corpus is ~43,473 words - fits in a single context window. You may not need a graph.

## Summary
- 602 nodes · 1749 edges · 20 communities (17 shown, 3 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 52 edges (avg confidence: 0.57)
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
- `graphify Query First Rule` --semantically_similar_to--> `Existing Graph Fast Path`  [INFERRED] [semantically similar]
  AGENTS.md → .codex/skills/graphify/SKILL.md
- `arctic-inference` --conceptually_related_to--> `Speculative Decoding`  [AMBIGUOUS]
  requirements.txt → README.md
- `Settings` --uses--> `DraftModelResolution`  [INFERRED]
  app.py → paths.py
- `Settings` --uses--> `CpuSuffixCache`  [INFERRED]
  app.py → suffix_cache.py
- `Settings` --uses--> `SuffixCacheConfig`  [INFERRED]
  app.py → suffix_cache.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Graphify Pipeline Flow** — _codex_skills_graphify_skill_full_pipeline, _codex_skills_graphify_skill_structural_extraction, _codex_skills_graphify_skill_semantic_extraction, _codex_skills_graphify_skill_build_cluster_outputs, _codex_skills_graphify_skill_graph_health_check [EXTRACTED 1.00]
- **Graphify Project Rules** — agents_graphify_project_rules, agents_graphify_query_first_rule, agents_dirty_graph_tolerance, agents_wiki_navigation_rule, agents_post_modify_update_rule [EXTRACTED 1.00]
- **vLLM JIT JetSpec Service Capabilities** — readme_vllm_jit_jetspec_service, readme_speculative_decoding, readme_manager_triggered_jetspec_training, readme_hub_checkpoint_sync, readme_live_sft_dpo_lora_training, readme_jit_status_endpoints, readme_rust_suffix_backend [EXTRACTED 1.00]

## Communities (20 total, 3 thin omitted)

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
Cohesion: 0.11
Nodes (32): _accepted_tokens_by_position(), _append_text_to_content(), _configure_logging(), _content_contains_text(), _default_enable_thinking(), _effective_accept_depth(), _env_first(), _force_thinking_enabled() (+24 more)

### Community 4 - "Settings And Cache"
Cohesion: 0.12
Nodes (8): _custom_proposer_env_name(), _messages_from_stored_prompt(), Settings, TrafficRecorder, VllmProcess, AsyncClient, MissingTrainingDependency, RuntimeError

### Community 5 - "Training Endpoints"
Cohesion: 0.16
Nodes (24): _active_live_lora_adapter_path(), _count_jsonl_rows(), _effective_training_snapshot(), health(), jit_train_data(), jit_train_jetspec(), jit_train_live_lora(), jit_train_once() (+16 more)

### Community 6 - "Rust Suffix Backend"
Cohesion: 0.14
Nodes (14): Bound, HashMap, Option, PyModule, PyResult, filter_jetspec_drafts(), merge_and_summarize_hybrid_drafts(), merge_hybrid_drafts() (+6 more)

### Community 7 - "Proxy Sample Extraction"
Cohesion: 0.14
Nodes (21): _completion_payload(), _content_text(), _env_json_obj(), _extract_completion(), _extract_prompt(), _extract_training_samples(), jit_rebuild_suffix_cache(), jit_sleep() (+13 more)

### Community 8 - "Hub Checkpoints"
Cohesion: 0.24
Nodes (22): _atomic_copy(), _atomic_write_json(), _checkpoint_name_from_latest(), _checkpoint_path_from_latest(), _checkpoint_step(), config_from_env(), download_newer_checkpoint(), _downloaded_latest_payload() (+14 more)

### Community 9 - "Graphify Workflow"
Cohesion: 0.13
Nodes (23): /graphify add URL, Exports Reference, Extraction Subagent Prompt, graphify merge-graphs, Post-Commit Hook, Constrained Query Expansion, Query Flow, Video Audio Transcription (+15 more)

### Community 10 - "Runtime Configuration"
Cohesion: 0.19
Nodes (17): _bool_cli_option(), _default_ephemeral_data_dir(), _env_bool_any(), _env_first_str(), _env_json_obj_any(), _extra_arg_present(), _live_lora_serving_enabled(), _live_trainer_parallel_mode() (+9 more)

### Community 11 - "Service Documentation"
Cohesion: 0.14
Nodes (19): Hub Checkpoint Sync, HybridSuffixJetSpecProposer, JetSpec Mode, /jit Status Endpoints, Live SFT/DPO LoRA Training, Manager-Triggered JetSpec Training, Plugin Hybrid Mode, Raw-Torch LoRA Trainer (+11 more)

### Community 12 - "Live Training Proxy"
Cohesion: 0.16
Nodes (17): _append_live_training_rows(), _filtered_request_headers(), _filtered_response_headers(), _ingest_live_training_rows(), jit_train_dpo(), jit_train_sft(), _live_lora_proxy_enabled(), _live_training_counts() (+9 more)

### Community 13 - "Speculative Config"
Cohesion: 0.19
Nodes (10): _auto_spec_method(), _custom_hybrid_speculative_config(), _env_bool(), _env_float(), _env_int(), _env_int_any(), _load_speculative_config(), PathLike (+2 more)

### Community 14 - "Checkpoint Paths"
Cohesion: 0.45
Nodes (11): checkpoint_dir_for_model(), checkpoint_dir_from_env(), _env_first_nonempty(), _env_nonempty(), latest_checkpoint_json_from_env(), _looks_like_hf_model_dir(), native_jetspec_draft_model_resolution_from_env(), _native_jetspec_local_candidates() (+3 more)

### Community 15 - "Trainer Lifecycle"
Cohesion: 0.32
Nodes (3): _env_float_any(), jit_wake(), shutdown()

### Community 17 - "Torch Defaults"
Cohesion: 0.83
Nodes (3): _configure_torch_defaults(), _running_pip(), _torch_float32_matmul_precision()

## Ambiguous Edges - Review These
- `Speculative Decoding` → `arctic-inference`  [AMBIGUOUS]
  requirements.txt · relation: conceptually_related_to

## Knowledge Gaps
- **10 isolated node(s):** `LoRALinear`, `vllm-dflash-jit-rust`, `graphify Skill Documentation`, `Graph Health Check`, `Extraction Subagent Prompt` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Speculative Decoding` and `arctic-inference`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `CpuSuffixCache` connect `Hybrid Proposer` to `Streaming Capture`, `App Request Handling`, `Settings And Cache`, `Training Endpoints`?**
  _High betweenness centrality (0.170) - this node is a cross-community bridge._
- **Why does `run_training()` connect `Live LoRA Trainer` to `Settings And Cache`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Why does `SuffixCacheConfig` connect `Hybrid Proposer` to `Streaming Capture`, `App Request Handling`, `Settings And Cache`, `Training Endpoints`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `CpuSuffixCache` (e.g. with `Settings` and `StreamingCompletionCapture`) actually correct?**
  _`CpuSuffixCache` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `JIT JetSpec speculation helpers for a vLLM development service.`, `Experimental vLLM custom proposer that combines CPU suffix and JetSpec drafts.`, `Small causal draft adapter used for JIT JetSpec experiments.      Native JetSpec` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Live LoRA Trainer` be split into smaller, more focused modules?**
  _Cohesion score 0.05836713101745724 - nodes in this community are weakly interconnected._