# vLLM Colocate: Inference + LoRA Training in One Container

One container, one base model, two GPU halves:

- **Inference**: vLLM serves the base model directly on the public inference
  port (default `8000`) using half of the visible GPUs. There is no proxy in
  front of it — clients talk straight to vLLM's OpenAI-compatible API.
- **Training**: the other half of the GPUs runs LoRA training jobs (SFT, DPO,
  or KTO) submitted through a small control-plane API (default `8001`).
  Training samples are provided in the request body — traces and training data
  are generated elsewhere; this service only trains and serves.
- **Fast weight sync**: after each training job finishes (or is stopped with
  checkpoints written), the exported PEFT adapter is loaded into the running
  vLLM server via `/v1/load_lora_adapter`. No restart, no model reload —
  inference requests keep flowing the whole time.

Because inference and training own disjoint GPUs, inference requests and
training jobs run concurrently. There is no sleep/wake handoff and no request
queueing during training.

## Architecture

```
                       ┌────────────────────────────────────────────┐
 inference clients ───►│ :8000  vllm serve (GPUs 0..N/2-1)          │
                       │        --enable-lora, runtime LoRA updates │
                       │              ▲                             │
                       │              │ /v1/load_lora_adapter       │
 training clients ────►│ :8001  control plane (FastAPI)             │
                       │        └► lora_trainer.py subprocess       │
                       │           (GPUs N/2..N-1, raw-torch LoRA)  │
                       └────────────────────────────────────────────┘
```

The trainer is a self-contained raw-torch process (`lora_trainer.py`): it
loads the same checkpoint vLLM serves through Transformers, injects LoRA
modules, trains with SFT/DPO/KTO losses, and exports a PEFT-format adapter
directory (`adapter_config.json` + `adapter_model.safetensors`) that vLLM can
load at runtime. DPO and KTO reference log-probabilities come from the same
model with LoRA disabled — no second model copy is needed. Jobs run
sequentially; each job resumes from the latest adapter checkpoint, so training
is cumulative across jobs.

## Run

```shell
docker run --rm --gpus all --ipc=host --shm-size 16g \
  -p 8000:8000 -p 8001:8001 \
  -v vllm-colocate-data:/data/vllm_colocate \
  -v hf-cache:/root/.cache/huggingface \
  -e HF_TOKEN \
  -e VLLM_COLOCATE_MODEL=Qwen/Qwen3-8B \
  <image>
```

Or from the official vLLM image without building:

```shell
docker run --rm --gpus all --ipc=host --shm-size 16g \
  -p 8000:8000 -p 8001:8001 \
  -v vllm-colocate-data:/data/vllm_colocate \
  -v hf-cache:/root/.cache/huggingface \
  -e HF_TOKEN \
  -e VLLM_COLOCATE_MODEL=Qwen/Qwen3-8B \
  --entrypoint /bin/bash \
  vllm/vllm-openai:latest \
  -lc 'git clone --depth 1 --branch main https://github.com/flozi00/vllm-self-spec.git /opt/vllm_colocate &&
       python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
         "fastapi>=0.115.0" "httpx>=0.28.0" "uvicorn>=0.34.0" \
         "huggingface_hub>=0.20.0" "safetensors>=0.4.0" &&
       export PYTHONPATH=/opt &&
       exec python -m vllm_colocate.app'
```

## GPU partitioning

By default the visible GPUs are split in half: inference gets the first half
(rounded up), training the rest. On an 8-GPU host, vLLM runs with
`--tensor-parallel-size 4` on GPUs `0-3` and training uses GPUs `4-7`.
Override the split explicitly:

```shell
VLLM_COLOCATE_INFERENCE_GPUS=0,1
VLLM_COLOCATE_TRAINING_GPUS=2,3
VLLM_COLOCATE_TENSOR_PARALLEL_SIZE=2
```

Tensor parallelism defaults to the largest power of two that fits the
inference half (attention-head counts rarely divide by 3), so a 6-GPU host
gets 3 inference GPUs with TP 2 unless you override it. Explicit GPU lists
must use the same identifiers the container sees (indices from `nvidia-smi`,
or the tokens of `CUDA_VISIBLE_DEVICES` if that is set); unknown ids fail at
startup instead of silently overlapping the two halves.

On a single-GPU host both roles share the device; GPU memory utilization for
vLLM then defaults to `0.45` instead of `0.90` so training has headroom.
Multi-GPU training uses Transformers `device_map=auto` model parallelism
inside the trainer subprocess automatically.

## Inference

Plain vLLM, served directly:

```shell
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "Hi"}]}'
```

The trained adapter is served under its own model name (default
`<served-model-name>-lora`). After the first weight sync:

```shell
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "Qwen/Qwen3-8B-lora", "messages": [{"role": "user", "content": "Hi"}]}'
```

Requesting the base model name always hits the untuned weights; requesting the
adapter name hits the latest synced adapter. `GET :8000/v1/models` lists both.

## Training

Submit jobs with samples inline. Each request creates one job; jobs queue and
run one at a time on the training GPUs. Add `?wait=true` (optionally with
`wait_timeout_seconds=...`) to block until the job finishes.

SFT — `prompt`+`completion`, chat `messages` (last assistant turn is the
target), or raw `text`:

```shell
curl -X POST http://127.0.0.1:8001/train/sft \
  -H 'content-type: application/json' \
  -d '{"samples": [{"prompt": "Question", "completion": "Answer"}]}'
```

DPO — `prompt` (or `messages`) with `chosen` and `rejected`:

```shell
curl -X POST http://127.0.0.1:8001/train/dpo \
  -H 'content-type: application/json' \
  -d '{"samples": [{"prompt": "Question", "chosen": "Better", "rejected": "Worse"}]}'
```

KTO — `prompt` (or `messages`), `completion`, and a boolean `label`
(`true` = desirable, `false` = undesirable; `desirable` is accepted as an
alias):

```shell
curl -X POST http://127.0.0.1:8001/train/kto \
  -H 'content-type: application/json' \
  -d '{"samples": [
        {"prompt": "Question", "completion": "Good answer", "label": true},
        {"prompt": "Question", "completion": "Bad answer", "label": false}
      ]}'
```

Per-job hyperparameter overrides go in `options`:

```json
{
  "samples": [...],
  "options": {"max_steps": 50, "learning_rate": 1e-4, "kto_beta": 0.1}
}
```

Allowed options: `max_steps`, `train_epochs`, `batch_size`,
`gradient_accumulation_steps`, `max_seq_len`, `learning_rate`,
`weight_decay`, `max_grad_norm`, `dpo_beta`, `kto_beta`,
`kto_desirable_weight`, `kto_undesirable_weight`, `loss_vocab_sample_size`.

Job lifecycle:

```shell
curl http://127.0.0.1:8001/train/jobs                  # recent jobs
curl http://127.0.0.1:8001/train/jobs/<job-id>          # one job
curl -X POST http://127.0.0.1:8001/train/jobs/<job-id>/cancel
curl http://127.0.0.1:8001/train/metrics?limit=50       # trainer step metrics
```

Job statuses: `queued`, `running`, `succeeded`, `failed`, `canceled`. A
canceled or failed job that still wrote checkpoints has its partial progress
synced (the checkpoint step advanced, so the adapter export is valid).

## Weight sync

After every job whose checkpoint step advanced, the control plane loads the
exported adapter into vLLM: unload the stable adapter name, then load the new
versioned adapter directory under the same name. Adapter exports are
versioned (`adapter-step-<n>/`), so a load never races a directory being
rewritten. The active adapter is re-synced automatically after a vLLM restart.

```shell
curl http://127.0.0.1:8001/adapter          # current adapter state
curl -X POST http://127.0.0.1:8001/adapter/sync   # force a re-sync
```

## Observability

```shell
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/status
```

`/status` reports the GPU partition, vLLM process/readiness state, loaded
model names, the training queue, the latest checkpoint, and the active
adapter. vLLM's own metrics stay on the inference port (`:8000/metrics`).

## Configuration reference

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_COLOCATE_MODEL` | `Qwen/Qwen3-8B` | Base model to serve and train |
| `VLLM_COLOCATE_SERVED_MODEL_NAME` | model name | Public model name in vLLM |
| `VLLM_COLOCATE_LORA_ADAPTER_NAME` | `<served>-lora` | Adapter model name |
| `VLLM_COLOCATE_INFERENCE_PORT` | `8000` | vLLM port |
| `VLLM_COLOCATE_API_PORT` | `8001` | Control-plane port |
| `VLLM_COLOCATE_INFERENCE_GPUS` | first half | Explicit inference GPU ids |
| `VLLM_COLOCATE_TRAINING_GPUS` | second half | Explicit training GPU ids |
| `VLLM_COLOCATE_TENSOR_PARALLEL_SIZE` | #inference GPUs | vLLM tensor parallelism |
| `VLLM_COLOCATE_GPU_MEMORY_UTILIZATION` | `0.90` (`0.45` shared) | vLLM GPU memory fraction |
| `VLLM_COLOCATE_MAX_MODEL_LEN` | model default | vLLM `--max-model-len` |
| `VLLM_COLOCATE_MAX_NUM_SEQS` | vLLM default | vLLM `--max-num-seqs` |
| `VLLM_COLOCATE_MAX_LORAS` | `1` | vLLM `--max-loras` |
| `VLLM_COLOCATE_MAX_LORA_RANK` | `max(16, LORA_R)` rounded up to a rank vLLM accepts | vLLM `--max-lora-rank` |
| `VLLM_COLOCATE_VLLM_EXTRA_ARGS` | empty | Extra `vllm serve` args |
| `VLLM_COLOCATE_DATA_DIR` | `/data/vllm_colocate` | Jobs, checkpoints, metrics |
| `VLLM_COLOCATE_LORA_R` / `_LORA_ALPHA` / `_LORA_DROPOUT` | `16` / `32` / `0.05` | LoRA shape |
| `VLLM_COLOCATE_LORA_TARGET_MODULES` | attn+mlp projections | LoRA injection targets |
| `VLLM_COLOCATE_TRAIN_STEPS` / `_TRAIN_EPOCHS` | `0` / `1` | Default steps (0 = full pass) |
| `VLLM_COLOCATE_BATCH_SIZE` / `_GRADIENT_ACCUMULATION_STEPS` | `1` / `1` | Batch shape |
| `VLLM_COLOCATE_MAX_SEQ_LEN` | `2048` | Trainer sequence length |
| `VLLM_COLOCATE_LR` | `2e-4` | Learning rate |
| `VLLM_COLOCATE_DPO_BETA` | `0.1` | DPO beta |
| `VLLM_COLOCATE_KTO_BETA` | `0.1` | KTO beta |
| `VLLM_COLOCATE_KTO_DESIRABLE_WEIGHT` / `_KTO_UNDESIRABLE_WEIGHT` | `1.0` / `1.0` | KTO loss weights |
| `VLLM_COLOCATE_TRAIN_QUANTIZATION` | `auto` | NVFP4/FP-quant training support |
| `VLLM_COLOCATE_TRUST_REMOTE_CODE` | `true` | Trust remote code |
| `VLLM_COLOCATE_SYNC_ON_STARTUP` | `true` | Re-sync adapter after vLLM (re)start |
| `VLLM_COLOCATE_CANCEL_GRACE_SECONDS` | `30` | Cancel grace before SIGTERM |
| `VLLM_COLOCATE_INFERENCE_HOST` / `_API_HOST` | `0.0.0.0` | Bind hosts for vLLM / control plane |
| `VLLM_COLOCATE_CHECKPOINT_DIR` | `<data>/checkpoints/<model>` | Checkpoint + adapter export dir |
| `VLLM_COLOCATE_JOBS_DIR` | `<data>/jobs` | Per-job sample JSONL dir |
| `VLLM_COLOCATE_METRICS_PATH` | `<data>/trainer_metrics.jsonl` | Trainer step-metrics file |
| `VLLM_COLOCATE_READY_TIMEOUT_SECONDS` | `1800` | vLLM readiness timeout |
| `VLLM_COLOCATE_WEIGHT_DECAY` / `_MAX_GRAD_NORM` | `0` / `1` | Optimizer defaults (per-job overridable) |
| `VLLM_COLOCATE_TORCH_DTYPE` | `bfloat16` | Trainer compute dtype |
| `VLLM_COLOCATE_CHECKPOINT_EVERY` | `16` | Steps between adapter exports (sync granularity) |
| `VLLM_COLOCATE_KEEP_LAST_CHECKPOINTS` | `2` | Checkpoint/adapter versions retained |
| `VLLM_COLOCATE_SAVE_OPTIMIZER_STATE` | `true` | Save optimizer state in checkpoints |
| `VLLM_COLOCATE_LOG_LEVEL` | `INFO` | Log level (supervisor and trainer) |

Quantized base checkpoints (ModelOpt/NVFP4 packed weights) are supported in
training through a raw-torch dequantization fallback, controlled by
`VLLM_COLOCATE_TRAIN_QUANTIZATION` (`auto` enables it when the model name
contains `nvfp4`).

## Notes and limits

- The DPO/KTO reference distribution is the base model (adapter disabled),
  also when a job resumes from an earlier adapter. Both losses use summed
  completion log-probabilities, matching the DPO/KTO papers and TRL.
- One adapter is trained cumulatively; per-job `lora_r` changes are not
  supported (the adapter shape is fixed by the environment).
- Training data is job-scoped: samples are stored under
  `VLLM_COLOCATE_DATA_DIR/jobs/<job-id>.jsonl` for reproducibility, but no
  traffic or message tracking of inference requests happens anywhere.
