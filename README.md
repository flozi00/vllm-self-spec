# vLLM JIT JetSpec Service

This wrapper is now intentionally narrow:

- launch `vllm serve` behind the existing OpenAI-compatible proxy
- enable low-latency speculative decoding with suffix decoding, JetSpec, or the hybrid plugin
- record real inference traffic and token traces for suffix reuse
- expose explicit routes for manager-triggered JetSpec and live LoRA training
- accept live SFT/DPO samples for raw-torch LoRA training
- expose concise `/jit/*` status and metrics endpoints

It does not generate synthetic data, patch model code, patch vLLM kernels, or
install model-specific monkey patches. The mounted `sitecustomize.py` only sets
Torch runtime defaults for performance; by default it calls
`torch.set_float32_matmul_precision("high")` to enable TF32 matmul on supported
NVIDIA GPUs. Override it with:

```shell
VLLM_JETSPEC_TORCH_FLOAT32_MATMUL_PRECISION=highest|high|medium|off
```

## Speculation Modes

Plugin hybrid is the default low-latency path for live traffic:

```shell
VLLM_JETSPEC_SPEC_METHOD=plugin_hybrid
```

That maps to vLLM's `custom_class` speculative API:

```json
{
  "method": "custom_class",
  "model": "vllm_dflash_jit.hybrid_proposer.HybridSuffixJetSpecProposer",
  "num_speculative_tokens": 8
}
```

The hybrid proposer gives the CPU suffix cache the draft budget first, then uses
the latest local JetSpec adapter checkpoint to fill remaining draft slots. Stats
are written to `VLLM_JETSPEC_PROPOSER_STATS_PATH` and surfaced through
`/jit/speculation_metrics`.

Native suffix decoding remains available:

```shell
VLLM_JETSPEC_SPEC_METHOD=suffix
```

Native JetSpec tree drafting is available when the deployed vLLM image is the
JetSpec vLLM fork:

```shell
VLLM_JETSPEC_SPEC_METHOD=jetspec
```

The current JetSpec fork exposes its runtime through vLLM's `method: "dflash"`
speculative-config key, so this wrapper still emits that method name for native
JetSpec mode only.

## Draft Heads

Native JetSpec draft-head resolution is local-first:

1. `VLLM_JETSPEC_DRAFT_HEAD`, `VLLM_JETSPEC_DRAFT_MODEL`, or `JETSPEC_DRAFT_HEAD`
2. `VLLM_JETSPEC_DRAFT_HEAD_DIR` or `VLLM_JETSPEC_DRAFT_MODEL_DIR`
3. `<VLLM_JETSPEC_CHECKPOINT_ROOT>/<base-model-slug>/jetspec`
4. `VLLM_JETSPEC_PRETRAINED_DRAFT_HEAD` or `VLLM_JETSPEC_PRETRAINED_DRAFT_MODEL`
5. the built-in JetSpec model-zoo mapping for supported public heads

The built-in trainer writes the small local adapter `.pt` format used by
the plugin hybrid proposer. Native JetSpec draft-head training should be wired
through `VLLM_JETSPEC_TRAINER_COMMAND` so an external trainer can write a Hugging
Face draft-head directory.

## Manager-Triggered JetSpec Training

The wrapper never schedules training on its own, including when the GPU is idle.
Only an explicit API request to `/jit/train/jetspec` may start JetSpec training.
A manager proxy should decide when to pause inference, sleep vLLM, and trigger
JetSpec training:

```shell
curl -X POST 'http://127.0.0.1:<public-port>/jit/sleep'
curl -X POST 'http://127.0.0.1:<public-port>/jit/train/jetspec?wait=true'
curl -X POST 'http://127.0.0.1:<public-port>/jit/wake'
```

The trainer stores `training_state.json` under the model checkpoint directory
and skips unchanged data on later manually triggered cycles unless `force=true`
is passed.

## Hub Checkpoint Sync

Set `VLLM_JETSPEC_HF_CHECKPOINT_REPO=<namespace>/<repo>` to mirror promoted
JetSpec adapter checkpoints through Hugging Face Hub. The wrapper uses the
served-model slug as the repo path prefix by default, so the Hub checkpoint name
follows the public deployment contract rather than the backing base model. For
example, a service with `VLLM_JETSPEC_SERVED_MODEL_NAME=smolagent` writes
`smolagent/latest.json` and `smolagent/checkpoint-step-...pt`.

Only the promoted adapter checkpoint and a sanitized `latest.json` are synced.
Traffic files, live training rows, suffix caches, and `training_state.json` are
not uploaded. The Hub `latest.json` allowlist is:

```json
{
  "checkpoint": "checkpoint-step-00000042.pt",
  "checkpoint_path": "checkpoint-step-00000042.pt",
  "step": 42,
  "saved_at": 1710000000.0,
  "model_name": "base/model",
  "num_speculative_tokens": 8,
  "quality": {"exact_prefix_mean": 0.5},
  "hub_repo_id": "namespace/repo",
  "hub_path_prefix": "smolagent",
  "hub_revision": "main",
  "uploaded_at": 1710000001.0
}
```

Minimum config:

```shell
VLLM_JETSPEC_HF_CHECKPOINT_REPO=namespace/private-spec-checkpoints
HF_TOKEN=<token with read/write access>
```

Defaults when the repo is set:

```shell
VLLM_JETSPEC_HF_CHECKPOINT_REVISION=main
VLLM_JETSPEC_HF_CHECKPOINT_DOWNLOAD=true
VLLM_JETSPEC_HF_CHECKPOINT_UPLOAD=true
VLLM_JETSPEC_HF_CHECKPOINT_SYNC_ON_INFERENCE=true
VLLM_JETSPEC_HF_CHECKPOINT_SYNC_INTERVAL_SECONDS=300
VLLM_JETSPEC_HF_CHECKPOINT_CREATE_REPO=false
VLLM_JETSPEC_HF_CHECKPOINT_REPO_PRIVATE=true
```

Only set `VLLM_JETSPEC_HF_CHECKPOINT_PATH_PREFIX` when the served model name is
not the Hub folder name you want.

Download checks run before `/jit/train/jetspec`, before `/jit/sleep`, and in the
background during inference no more often than the configured interval. After a
successful promoted training checkpoint, the wrapper uploads the `.pt` first and
then uploads `latest.json`, so other nodes only see a new version after its
weights are present. Operators can force a sync explicitly:

```shell
curl -X POST 'http://127.0.0.1:<public-port>/jit/checkpoints/sync?direction=download&force=true'
curl -X POST 'http://127.0.0.1:<public-port>/jit/checkpoints/sync?direction=upload'
```

Useful knobs:

```shell
VLLM_JETSPEC_TRAINING_ENABLED=false
VLLM_JETSPEC_TRAIN_STEPS=0
VLLM_JETSPEC_TRAIN_EPOCHS=1
VLLM_JETSPEC_MAX_EXAMPLES_PER_TRAFFIC=0
```

`VLLM_JETSPEC_TRAINING_ENABLED=false` keeps legacy idle schedulers disabled;
the explicit `/jit/train/jetspec` API route still owns JetSpec training.
`VLLM_JETSPEC_TRAIN_STEPS=0` auto-sizes each route-triggered run to a full pass
over the built examples. While an explicit trainer route owns the GPUs,
OpenAI-compatible inference requests return a retryable `503` instead of
implicitly stopping the trainer. Use `/jit/wake` or stop the trainer route from
the manager if inference should take priority.

## Live SFT/DPO LoRA Training

Live model training uses durable JSONL at
`VLLM_JETSPEC_LIVE_TRAIN_DATA_PATH` and a separate
`<checkpoint-dir>/live_lora` checkpoint tree. It does not use TRL, VERL, or a
trainer framework: the built-in trainer uses Transformers only as the
autograd-capable PyTorch model/remote-code loader for the same checkpoint that
vLLM serves. It injects LoRA modules in raw torch, supports SFT and DPO losses,
and defaults to model-parallel loading for tensor-parallel-served models. The
running vLLM model is not mutated in-place; vLLM's inference workers do not
expose a backward/optimizer API through the OpenAI or tokenizer routes.

For quantized serving checkpoints, keep `VLLM_JETSPEC_MODEL` pointed at the
served model. The live trainer fine-tunes a LoRA adapter against that same
checkpoint and includes a raw-torch ModelOpt/NVFP4 fallback for packed expert
weights whose on-disk tensors are stored as two FP4 values per byte.

Append SFT samples:

```shell
curl -X POST http://127.0.0.1:<public-port>/jit/train/sft \
  -H 'content-type: application/json' \
  -d '{"samples":[{"prompt":"Question","completion":"Answer"}]}'
```

Append DPO samples:

```shell
curl -X POST http://127.0.0.1:<public-port>/jit/train/dpo \
  -H 'content-type: application/json' \
  -d '{"samples":[{"prompt":"Question","chosen":"Better answer","rejected":"Worse answer"}]}'
```

Trigger LoRA training explicitly after appending samples:

```shell
curl -X POST 'http://127.0.0.1:<public-port>/jit/sleep'
curl -X POST 'http://127.0.0.1:<public-port>/jit/train/live_lora?wait=true'
curl -X POST 'http://127.0.0.1:<public-port>/jit/wake'
```

While live LoRA training is running, inference requests are rejected with a
retryable `503` response. The manager proxy should drain or reroute users before
calling `/jit/train/live_lora`; the wrapper does not decide when idle training
should happen.

When
`VLLM_JETSPEC_LIVE_LORA_SERVING_ENABLED=true`, the active adapter is loaded into
vLLM through its LoRA endpoint and the proxy rewrites public requests for the
served model name to the active adapter name, preserving the client-facing model
contract. Keep that serving flag off for vLLM/model/quantization combinations
that do not initialize with `--enable-lora`; training still checkpoints the
adapter for later promotion.

Useful knobs:

```shell
VLLM_JETSPEC_LIVE_TRAINING_ENABLED=true
VLLM_JETSPEC_LIVE_LORA_SERVING_ENABLED=false
VLLM_JETSPEC_LIVE_TRAIN_PARALLEL_MODE=auto
VLLM_JETSPEC_LIVE_TRAIN_QUANTIZATION=auto
VLLM_JETSPEC_LIVE_LORA_R=16
VLLM_JETSPEC_LIVE_INCLUDE_EXPERT_LORA=false
VLLM_JETSPEC_LIVE_MAX_SEQ_LEN=2048
VLLM_JETSPEC_LIVE_LOSS_VOCAB_SAMPLE_SIZE=0
```

`loss_vocab_sample_size` can also be passed to `/jit/train/live_lora`; `0` uses
the exact full-vocabulary loss, while a positive value uses a deterministic
sampled-token loss that is useful for tiny route smoke tests on very large
vocabularies.

## Observability

```shell
curl http://127.0.0.1:<public-port>/jit/status
curl http://127.0.0.1:<public-port>/jit/training_metrics?limit=50
curl http://127.0.0.1:<public-port>/jit/speculation_metrics
curl http://127.0.0.1:<public-port>/jit/train/data
```

`/jit/status` includes traffic row counts, external scheduler state, latest
trainer progress, latest checkpoint metadata, suffix-cache state, and vLLM
speculative metrics when the runtime exposes them.

## Rust Suffix Backend

The hot suffix proposal loop can use the optional Rust/PyO3 backend. If the
extension is absent, the Python implementation is used.

```shell
cd services/vllm_dflash_jit/rust_suffix_backend
cargo build --release
cp target/release/lib_vllm_dflash_jit_rust.so ../_vllm_dflash_jit_rust.so
```
