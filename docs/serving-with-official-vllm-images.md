# Serving with the official vLLM containers

This plugin does not need an image of its own. It is three Python files that
run *inside* the stock `vllm/vllm-openai` image: the launcher imports vLLM's
own OpenAI server, attaches `POST /train/sft` to the same FastAPI app, and
hands control to vLLM's `serve` entrypoint. Everything else — the engine, the
CUDA stack, the OpenAI routes — is the official image, untouched.

For Kubernetes, see [`kubernetes.md`](kubernetes.md); it applies the same
approach through manifests.

## Why the stock image is enough

Two things usually force a custom image, and neither applies here.

**No extra dependencies.** The plugin imports `fastapi`, `httpx`,
`huggingface_hub`, `safetensors`, `transformers` and `torch`. Current
`vllm/vllm-openai` images already ship all of them — `transformers`,
`huggingface_hub` and `safetensors` are direct vLLM requirements, and `fastapi`
plus `httpx` arrive with vLLM's `fastapi[standard]` dependency. Check any image
you plan to use:

```shell
docker run --rm --entrypoint python vllm/vllm-openai:latest \
  -c 'import fastapi, httpx, huggingface_hub, safetensors, transformers, torch; print("ok")'
```

If that prints `ok`, the plugin needs nothing but its own source, and none of
the commands below install anything. On an older image where the check fails,
add the missing packages the way the repository's `Dockerfile` does — with
`pip install --upgrade-strategy only-if-needed`, which is what keeps pip from
replacing the image's CUDA-matched torch build.

**No patching.** The in-flight counter that gates training is installed through
vLLM's own `--middleware` flag, and the extra route is added to the app vLLM
builds. There is no fork and no monkey-patching of engine internals, so an
image upgrade is just a tag change.

## What the launcher changes about `vllm serve`

`python -m vllm_colocate.app` behaves like `vllm serve` with the same flags,
with these differences:

| | Effect |
| --- | --- |
| `--enable-lora` | Always on — the trained adapter is served as a LoRA |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` | Set before vLLM is imported, so adapters can be hot-swapped |
| `--api-server-count` | Forced to 1; the route, the idle gate and runtime LoRA all need the single in-process API server |
| `--gpu-memory-utilization` | Defaults to `0.45` instead of vLLM's own near-full default, leaving VRAM for the trainer |
| `--max-lora-rank`, `--max-loras` | Derived from the LoRA settings unless you pass them |
| `--headless`, `--uds` | Rejected: no API server to attach to, and the self-calls need local TCP |

Explicit flags always win. Ordering is launcher defaults, then
`VLLM_COLOCATE_VLLM_EXTRA_ARGS`, then the process's own command line — so
anything you pass on the container command line overrides the rest:

```shell
python -m vllm_colocate.app Qwen/Qwen3-8B --max-model-len 8192 --port 9000
```

## Getting the source into the container

The package directory must be named `vllm_colocate` and its parent must be on
`PYTHONPATH`. Pick whichever delivery fits your environment.

### 1. Bind mount a checkout (development)

Fastest loop: edit on the host, restart the container.

```shell
git clone https://github.com/flozi00/vllm-self-spec.git

docker run --rm --gpus all --ipc=host --shm-size 16g \
  -p 8000:8000 \
  -v "$PWD/vllm-self-spec:/opt/vllm_colocate:ro" \
  -v vllm-colocate-data:/data/vllm_colocate \
  -v hf-cache:/root/.cache/huggingface \
  -e PYTHONPATH=/opt \
  -e HF_TOKEN \
  -e VLLM_COLOCATE_MODEL=Qwen/Qwen3-8B \
  --entrypoint python \
  vllm/vllm-openai:latest \
  -m vllm_colocate.app
```

`--entrypoint python` replaces the image's `ENTRYPOINT ["vllm", "serve"]`; the
arguments after the image name are what `python` runs.

### 2. Fetch at container start (no local checkout)

One self-contained command — useful for a throwaway box or a CI job:

```shell
docker run --rm --gpus all --ipc=host --shm-size 16g \
  -p 8000:8000 \
  -v vllm-colocate-data:/data/vllm_colocate \
  -v hf-cache:/root/.cache/huggingface \
  -e HF_TOKEN \
  -e VLLM_COLOCATE_MODEL=Qwen/Qwen3-8B \
  --entrypoint /bin/bash \
  vllm/vllm-openai:latest \
  -lc 'mkdir -p /opt/vllm_colocate &&
       curl -fsSL https://github.com/flozi00/vllm-self-spec/archive/refs/heads/main.tar.gz |
         tar -xz --strip-components=1 -C /opt/vllm_colocate &&
       export PYTHONPATH=/opt &&
       exec python -m vllm_colocate.app'
```

Note the tarball rather than `git clone`: the official runtime image installs
`curl` but not `git`, so a clone fails with `git: command not found`. Swap
`refs/heads/main` for `refs/tags/<tag>` — or a commit SHA under
`archive/<sha>.tar.gz` — for anything you intend to run twice the same way.
This needs egress to GitHub on every start; if that is not acceptable, use
option 1 or 3.

### 3. Build the thin image (production)

The repository's `Dockerfile` is a copy of the three files on top of the
official image, nothing more. Build once, run anywhere, no egress at start:

```shell
TAG="registry.example.com/vllm-colocate:$(git rev-parse --short HEAD)"
docker build -t "$TAG" .
docker push "$TAG"
```

Pin the base while you are at it:

```shell
docker build --build-arg VLLM_BASE_IMAGE=vllm/vllm-openai:vX.Y.Z -t ... .
```

## Runtime essentials

**Ports.** One port, default `8000`: OpenAI routes and `POST /train/sft` share
it. There is no second port to expose.

**Shared memory.** `--ipc=host --shm-size 16g` (or at minimum a generous
`--shm-size`). vLLM's workers communicate over `/dev/shm`; the container
default of 64 MB is not enough.

**GPUs.** `--gpus all`. Inference and training share every visible GPU;
tensor parallelism defaults to the largest power of two that fits the GPU count
(so 4 GPUs give TP=4, 6 GPUs give TP=4). Set
`VLLM_COLOCATE_TENSOR_PARALLEL_SIZE` to override.

**Volumes.** Two directories are worth persisting:

| Path | Contents | Losing it means |
| --- | --- | --- |
| `/data/vllm_colocate` | `checkpoints/<model>/` (LoRA state, exported adapters, `latest.json`), `jobs/*.jsonl`, `trainer_metrics.jsonl`, `trainer.pause` | Everything the model has been taught |
| `~/.cache/huggingface` (or `HF_HOME`) | Base weights | Re-downloading the model on every start |

The base model itself is never modified; training only ever produces adapters
under the checkpoint directory.

**Config.** Everything is environment variables — the full table lives in the
[README](../README.md#configuration-reference).

## Verify

```shell
curl -fsS localhost:8000/health && echo healthy
curl -fsS localhost:8000/v1/models | jq -r '.data[].id'

curl -fsS localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Hi"}]}'

# Queue a training job and block until it finishes.
curl -fsS -X POST 'localhost:8000/train/sft?wait=true' \
  -H 'content-type: application/json' \
  -d '{"samples":[{"prompt":"Who runs this service?","completion":"You do."}]}'

# The adapter is now a second model name.
curl -fsS localhost:8000/v1/models | jq -r '.data[].id'
```

## Running as a non-root user

The published `vllm/vllm-openai` image runs as root, but it already contains a
`vllm` user (UID 2000, GID 0) with group-0-writable caches, so it also runs
under `--user 2000:0`:

```shell
docker run --rm --gpus all --ipc=host --shm-size 16g \
  --user 2000:0 \
  -e HF_HOME=/hf-cache \
  -v hf-cache:/hf-cache \
  ...
```

Two things to get right when you drop root. First, every writable path
(`VLLM_COLOCATE_DATA_DIR`, `HF_HOME`, `VLLM_CACHE_ROOT`) must be a volume that
UID can write — and Docker, unlike Kubernetes' `fsGroup`, does not adjust
volume ownership for you, so hand a fresh volume over once:

```shell
docker volume create hf-cache
docker run --rm -v hf-cache:/hf-cache alpine chown -R 2000:0 /hf-cache
```

Second, `pip install` into the image's site-packages will fail as a non-root
user — which is fine, because the dependency check above says you do not need
it.

## Version pinning

`latest` moves. Pin the image tag for anything that matters, and re-run the
dependency check when you bump it. The launcher already absorbs vLLM's own
entrypoint reshuffles — it looks for `vllm.entrypoints.launchers` first and
falls back to the older `vllm.entrypoints.openai.api_server` layout — so a
version bump is normally a tag change and a restart.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `No module named vllm_colocate` | The source is not at `$PYTHONPATH/vllm_colocate`, or the directory is named something else (it must be `vllm_colocate`) |
| Server starts as plain vLLM, `/train/sft` returns 404 | The image's default entrypoint ran; pass `--entrypoint python ... -m vllm_colocate.app` |
| `RuntimeError` about shared memory, or workers hanging at startup | `--shm-size` too small |
| CUDA OOM once training starts | `VLLM_COLOCATE_GPU_MEMORY_UTILIZATION` too high for the trainer's share, or `VLLM_COLOCATE_MAX_MODEL_LEN` left at a very large model default |
| Training jobs stay `queued` forever | Inference never goes idle for `VLLM_COLOCATE_IDLE_GRACE_SECONDS`; check `GET /train/sft` for `inference.busy` |
| Adapter trains but responses look untuned | Request the adapter model name (`<served-model-name>-lora`), not the base name |
