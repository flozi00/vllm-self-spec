# vLLM Colocate: Inference + SFT LoRA Training on One Port

Stock vLLM serving with one extra route. The launcher runs vLLM's own
OpenAI-compatible server in-process and attaches `POST /train/sft` to the same
app on the same port:

- **Inference**: plain `vllm serve` behavior on the public port (default
  `8000`) using **all** visible GPUs. No proxy, no second port, no GPU
  partitioning.
- **Training**: `POST /train/sft` queues an SFT LoRA job with the samples from
  the request body. Jobs run sequentially in a trainer subprocess that shares
  the GPUs with vLLM — and only steps **while inference is idle**.
- **No weight reloads**: the base model (quantized checkpoints included) stays
  resident in vLLM the whole time. After each job the exported PEFT adapter is
  hot-swapped via vLLM's runtime `/v1/load_lora_adapter`; vLLM applies the
  LoRA on top of the resident weights, which is mathematically the merged
  model without ever touching the base weights.

## Architecture

```
                 ┌──────────────────────────────────────────────────┐
 clients ───────►│ :8000  vLLM OpenAI API (all GPUs)                │
                 │   /v1/chat/completions, /v1/models, ...          │
                 │   /train/sft  ◄─ SFT samples in, job status out  │
                 │        │                                         │
                 │        ▼ queued jobs, one at a time              │
                 │   lora_trainer.py subprocess (same GPUs)         │
                 │        │  pauses while inference requests run    │
                 │        ▼ exported PEFT adapter                   │
                 │   /v1/load_lora_adapter (hot swap, no reload)    │
                 └──────────────────────────────────────────────────┘
```

How the idle gating works: an in-flight request counter wraps the vLLM app
(installed through vLLM's own `--middleware` mechanism). While inference
requests are running — or finished less than `VLLM_COLOCATE_IDLE_GRACE_SECONDS`
ago — a pause file exists; the trainer checks it between training steps and
waits. When traffic stops, training resumes automatically. Memory is shared
statically: vLLM's `--gpu-memory-utilization` defaults to `0.45` here so the
trainer has headroom on the same GPUs.

The trainer is a self-contained raw-torch process (`lora_trainer.py`): it
loads the same checkpoint vLLM serves through Transformers, injects LoRA
modules, trains with the SFT loss, and exports a PEFT-format adapter directory
(`adapter_config.json` + `adapter_model.safetensors`) that vLLM loads at
runtime. Jobs run sequentially; each job resumes from the latest adapter
checkpoint, so training is cumulative across jobs.

## Run

```shell
docker run --rm --gpus all --ipc=host --shm-size 16g \
  -p 8000:8000 \
  -v vllm-colocate-data:/data/vllm_colocate \
  -v hf-cache:/root/.cache/huggingface \
  -e HF_TOKEN \
  -e VLLM_COLOCATE_MODEL=Qwen/Qwen3-8B \
  <image>
```

Or from the official vLLM image without building:

```shell
docker run --rm --gpus all --ipc=host --shm-size 16g \
  -p 8000:8000 \
  -v vllm-colocate-data:/data/vllm_colocate \
  -v hf-cache:/root/.cache/huggingface \
  -e HF_TOKEN \
  -e VLLM_COLOCATE_MODEL=Qwen/Qwen3-8B \
  --entrypoint /bin/bash \
  vllm/vllm-openai:latest \
  -lc 'git clone --depth 1 --branch main https://github.com/flozi00/vllm-self-spec.git /opt/vllm_colocate &&
       python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
         "fastapi>=0.115.0" "httpx>=0.28.0" \
         "huggingface_hub>=0.20.0" "safetensors>=0.4.0" &&
       export PYTHONPATH=/opt &&
       exec python -m vllm_colocate.app'
```

Any extra command-line arguments are passed straight to vLLM's own `serve`
parser, so this behaves like `vllm serve` with one additional route:

```shell
python -m vllm_colocate.app Qwen/Qwen3-8B --max-model-len 8192 --port 9000
```

Explicit flags win over the launcher's defaults. The launcher pins the things
the training route needs: `--enable-lora` is always on, runtime LoRA updating
is enabled, and `--api-server-count` stays at 1 (the route, the idle gate, and
runtime LoRA all need the single in-process API server).

## Inference

Plain vLLM, served directly:

```shell
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "Hi"}]}'
```

The trained adapter is served under its own model name (default
`<served-model-name>-lora`). After the first training job:

```shell
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "Qwen/Qwen3-8B-lora", "messages": [{"role": "user", "content": "Hi"}]}'
```

Requesting the base model name always hits the untuned weights; requesting the
adapter name hits the latest trained adapter. `GET :8000/v1/models` lists both.

## Training

One route. `POST` submits a job, `GET` reports state.

SFT samples are `prompt`+`completion`, chat `messages` (the last assistant
turn is the target), or raw `text`:

```shell
curl -X POST http://127.0.0.1:8000/train/sft \
  -H 'content-type: application/json' \
  -d '{"samples": [{"prompt": "Question", "completion": "Answer"}]}'
```

Each request creates one job; jobs queue and run one at a time. Add
`?wait=true` (optionally with `wait_timeout_seconds=...`) to block until the
job finishes. The response is the job payload (`id`, `status`, checkpoint
steps, adapter sync result, ...).

Per-job hyperparameter overrides go in `options`:

```json
{
  "samples": [...],
  "options": {"max_steps": 50, "learning_rate": 1e-4}
}
```

Allowed options: `max_steps`, `train_epochs`, `batch_size`,
`gradient_accumulation_steps`, `max_seq_len`, `learning_rate`,
`weight_decay`, `max_grad_norm`, `loss_vocab_sample_size`.

Status — jobs, queue, latest checkpoint, active adapter, gate state, and the
trainer's recent step metrics:

```shell
curl http://127.0.0.1:8000/train/sft
curl 'http://127.0.0.1:8000/train/sft?limit=100'   # more job history
```

Job statuses: `queued`, `running`, `succeeded`, `failed`, `canceled` (server
shutdown). A failed job that still advanced its checkpoint has the partial
progress synced into vLLM.

## Weight sync

After every job whose checkpoint step advanced, the launcher loads the
exported adapter into the running server: unload the stable adapter name, then
load the new versioned adapter directory under the same name. Adapter exports
are versioned (`adapter-step-<n>/`), so a load never races a directory being
rewritten. If a load fails, the previous known-good adapter is rolled back so
inference keeps serving tuned weights. On startup the latest adapter is
restored automatically once the server reports healthy.

## Configuration reference

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_COLOCATE_MODEL` | `Qwen/Qwen3-8B` | Base model to serve and train |
| `VLLM_COLOCATE_SERVED_MODEL_NAME` | model name | Public model name in vLLM |
| `VLLM_COLOCATE_LORA_ADAPTER_NAME` | `<served>-lora` | Adapter model name |
| `VLLM_COLOCATE_HOST` / `VLLM_COLOCATE_PORT` | `0.0.0.0` / `8000` | Bind address of the single server |
| `VLLM_COLOCATE_GPU_MEMORY_UTILIZATION` | `0.45` | vLLM memory fraction (rest is training headroom) |
| `VLLM_COLOCATE_TENSOR_PARALLEL_SIZE` | largest power of two <= #GPUs | vLLM tensor parallelism |
| `VLLM_COLOCATE_IDLE_GRACE_SECONDS` | `5` | Inference must be idle this long before training resumes |
| `VLLM_COLOCATE_MAX_MODEL_LEN` | model default | vLLM `--max-model-len` |
| `VLLM_COLOCATE_MAX_NUM_SEQS` | vLLM default | vLLM `--max-num-seqs` |
| `VLLM_COLOCATE_MAX_LORAS` | `1` | vLLM `--max-loras` |
| `VLLM_COLOCATE_MAX_LORA_RANK` | `max(16, LORA_R)` rounded up to a rank vLLM accepts | vLLM `--max-lora-rank` |
| `VLLM_COLOCATE_VLLM_EXTRA_ARGS` | empty | Extra `vllm serve` args (CLI args work too) |
| `VLLM_COLOCATE_DATA_DIR` | `/data/vllm_colocate` | Jobs, checkpoints, metrics, pause file |
| `VLLM_COLOCATE_LORA_R` / `_LORA_ALPHA` / `_LORA_DROPOUT` | `16` / `32` / `0.05` | LoRA shape |
| `VLLM_COLOCATE_LORA_TARGET_MODULES` | attn+mlp projections | LoRA injection targets |
| `VLLM_COLOCATE_TRAIN_STEPS` / `_TRAIN_EPOCHS` | `0` / `1` | Default steps (0 = full pass) |
| `VLLM_COLOCATE_BATCH_SIZE` / `_GRADIENT_ACCUMULATION_STEPS` | `1` / `1` | Batch shape |
| `VLLM_COLOCATE_MAX_SEQ_LEN` | `2048` | Trainer sequence length |
| `VLLM_COLOCATE_LR` | `2e-4` | Learning rate |
| `VLLM_COLOCATE_WEIGHT_DECAY` / `_MAX_GRAD_NORM` | `0` / `1` | Optimizer defaults (per-job overridable) |
| `VLLM_COLOCATE_TRAIN_QUANTIZATION` | `auto` | NVFP4/FP-quant training support |
| `VLLM_COLOCATE_TORCH_DTYPE` | `bfloat16` | Trainer compute dtype |
| `VLLM_COLOCATE_TRUST_REMOTE_CODE` | `true` | Trust remote code |
| `VLLM_COLOCATE_SYNC_ON_STARTUP` | `true` | Restore latest adapter after startup |
| `VLLM_COLOCATE_CHECKPOINT_DIR` | `<data>/checkpoints/<model>` | Checkpoint + adapter export dir |
| `VLLM_COLOCATE_JOBS_DIR` | `<data>/jobs` | Per-job sample JSONL dir |
| `VLLM_COLOCATE_METRICS_PATH` | `<data>/trainer_metrics.jsonl` | Trainer step-metrics file |
| `VLLM_COLOCATE_PAUSE_FILE` | `<data>/trainer.pause` | Idle-gate pause file |
| `VLLM_COLOCATE_READY_TIMEOUT_SECONDS` | `1800` | Startup adapter-restore readiness timeout |
| `VLLM_COLOCATE_CHECKPOINT_EVERY` | `16` | Steps between adapter exports (sync granularity) |
| `VLLM_COLOCATE_KEEP_LAST_CHECKPOINTS` | `2` | Checkpoint/adapter versions retained |
| `VLLM_COLOCATE_SAVE_OPTIMIZER_STATE` | `true` | Save optimizer state in checkpoints |
| `VLLM_COLOCATE_LOG_LEVEL` | `INFO` | Log level (launcher and trainer) |

Quantized base checkpoints (ModelOpt/NVFP4 packed weights) are supported in
training through a raw-torch dequantization fallback, controlled by
`VLLM_COLOCATE_TRAIN_QUANTIZATION` (`auto` enables it when the model name
contains `nvfp4`). Serving stays on the quantized weights the whole time; the
trained LoRA rides on top of them.

## Notes and limits

- Training compute-shares the GPUs with inference: steps run only while the
  server has been idle for the grace period, and a request arriving mid-step
  is delayed at most by that one step's remaining microbatches.
- One adapter is trained cumulatively; per-job `lora_r` changes are not
  supported (the adapter shape is fixed by the environment).
- Training data is job-scoped: samples are stored under
  `VLLM_COLOCATE_DATA_DIR/jobs/<job-id>.jsonl` for reproducibility, but no
  traffic or message tracking of inference requests happens anywhere.
- Run with `python -m vllm_colocate.app` (or the Docker entrypoint). The
  in-flight middleware is imported by dotted path, so the file must be
  importable as a module.
