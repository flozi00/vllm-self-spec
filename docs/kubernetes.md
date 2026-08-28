# Running on Kubernetes

Manifests that run this plugin on the **official** `vllm/vllm-openai` image:
no custom image, no build step, no egress at pod start. They live in
[`deploy/kubernetes/`](../deploy/kubernetes) and are ready to apply.

The container-level story — why the stock image is enough, what the launcher
changes about `vllm serve`, how to run it under Docker — is in
[`serving-with-official-vllm-images.md`](serving-with-official-vllm-images.md).
This page is the Kubernetes-specific half.

## What you are deploying

One pod that serves inference and trains on the same GPUs, on one port:

```
   Service :8000 ──► pod vllm-colocate-0
                       ├─ vLLM OpenAI API      (/v1/..., all GPUs)
                       ├─ POST /train/sft      (same app, same port)
                       └─ trainer subprocess   (steps only while inference is idle)
                             │
                       PVC data ── checkpoints + exported LoRA adapters
                       PVC hf-cache ── base weights
```

### Why a single replica

`replicas: 1` is not a starting point to scale from; it is the supported
topology. The pod owns state that cannot be duplicated or shared:

- **Checkpoints are pod-local.** Each replica would train its own adapter from
  its own job queue, so identical requests would get different answers
  depending on which pod they land on.
- **The job queue is in-process.** `POST /train/sft` queues into the memory of
  the pod that received it. A Service in front of N pods spreads a training
  batch across N unrelated queues.
- **The data volume has one writer.** Checkpoint exports and `latest.json`
  assume a single trainer.
- **The API server is single-process by design.** The launcher forces
  `--api-server-count 1` and rejects `--headless`, because the training route,
  the idle gate and runtime LoRA updates all need the one in-process app. That
  also rules out the multi-node patterns (LeaderWorkerSet, Ray-based
  disaggregation) that scale plain vLLM beyond a node.

Scale *up* instead: a bigger node, more GPUs in the same pod, a larger
`--tensor-parallel-size`. If you need read-only replicas of a finished
adapter, copy the exported adapter directory out and serve it with an ordinary
`vllm serve --enable-lora` Deployment — those replicas are stateless and scale
normally.

## Prerequisites

- A GPU node pool with the NVIDIA device plugin (or GPU Operator) exposing
  `nvidia.com/gpu`, and **whole** GPUs — not MIG slices or time-slicing
  replicas. The trainer lives in the VRAM vLLM did not reserve, and neither
  sharing mode can guarantee that headroom.
- A StorageClass for `ReadWriteOnce` volumes. Prefer one with
  `reclaimPolicy: Retain` for the checkpoint volume.
- Nodes able to pull the vLLM image (it is large — pre-pulling makes rollouts
  much less exciting).
- A `HF_TOKEN` only if the model is gated or private.

## Quick start

```shell
# 1. namespace
kubectl apply -f deploy/kubernetes/namespace.yaml

# 2. the plugin itself: three files, straight from the checkout
kubectl -n vllm-colocate create configmap vllm-colocate-src \
  --from-file=__init__.py --from-file=app.py --from-file=lora_trainer.py

# 3. optional: token for gated models
kubectl -n vllm-colocate create secret generic huggingface \
  --from-literal=HF_TOKEN="$HF_TOKEN"

# 4. everything else
kubectl apply -k deploy/kubernetes/
```

Watch it come up — the first start downloads the weights, so give it time:

```shell
kubectl -n vllm-colocate get pods -w
kubectl -n vllm-colocate logs -f statefulset/vllm-colocate
```

Then talk to it:

```shell
kubectl -n vllm-colocate port-forward svc/vllm-colocate 8000:8000 &

curl -fsS localhost:8000/v1/models | jq -r '.data[].id'

curl -fsS -X POST 'localhost:8000/train/sft?wait=true&wait_timeout_seconds=600' \
  -H 'content-type: application/json' \
  -d '{"samples":[{"prompt":"Who runs this service?","completion":"You do."}]}'

curl -fsS localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B-lora","messages":[{"role":"user","content":"Who runs this service?"}]}'
```

## How the manifests fit together

### Source delivery: a ConfigMap

The plugin is `__init__.py`, `app.py` and `lora_trainer.py` — about 135 KB,
comfortably inside the ~1 MiB ConfigMap limit. The ConfigMap is mounted at
`/opt/vllm_colocate` and `PYTHONPATH=/opt` makes it importable, so the
container command is just:

```yaml
command: ["python", "-m", "vllm_colocate.app"]
```

which replaces the image's `ENTRYPOINT ["vllm", "serve"]`.

This is the default because it needs no registry, no build and no network at
pod start. The cost is that the ConfigMap is a snapshot: updating the plugin
means re-creating it and restarting the pod.

```shell
kubectl -n vllm-colocate create configmap vllm-colocate-src \
  --from-file=__init__.py --from-file=app.py --from-file=lora_trainer.py \
  --dry-run=client -o yaml | kubectl apply --server-side -f -

kubectl -n vllm-colocate rollout restart statefulset/vllm-colocate
```

(`--server-side` avoids the client-side apply annotation, which would store a
second copy of every file in the object.)

### Volumes

| Mount | Backing | Why |
| --- | --- | --- |
| `/opt/vllm_colocate` | ConfigMap `vllm-colocate-src` | The plugin source |
| `/data/vllm_colocate` | PVC `data` (volumeClaimTemplate) | Checkpoints, exported adapters, job files, metrics — everything the model has learned |
| `/hf-cache` | PVC `hf-cache` (volumeClaimTemplate) | Base weights; `HF_HOME` points here |
| `/dev/shm` | `emptyDir` with `medium: Memory` | vLLM's workers communicate over shared memory; the 64 MB container default is not enough. This is the `--shm-size` of `docker run`, and it is why you do **not** need `hostIPC` |

Pages in a memory-backed `emptyDir` are charged to the pod's memory cgroup as
they are used, so `/dev/shm` competes with the process for the container's
memory limit: 16Gi of shared memory under a 64Gi limit leaves 48Gi for
everything else once it fills up. Budget for both.

### Resources

```yaml
requests: {cpu: "8", memory: 32Gi}
limits:   {nvidia.com/gpu: "1", memory: 64Gi}
```

- `nvidia.com/gpu` belongs in `limits`; Kubernetes copies it to requests. Whole
  GPUs only, for the reason in the prerequisites.
- The trainer is a **subprocess of this container**, so its host memory counts
  against the same limit. An OOM kill takes inference down with it — size for
  base weights plus optimizer state plus the CPU-side dequantization path, and
  set `requests == limits` if your platform wants Guaranteed QoS.
- More than one GPU: raise the `nvidia.com/gpu` limit. Tensor parallelism
  defaults to the largest power of two that fits (6 GPUs still means TP=4), so
  request GPUs in powers of two or set
  `VLLM_COLOCATE_TENSOR_PARALLEL_SIZE` explicitly.

### Probes

All three probes hit `GET /health`, vLLM's own liveness endpoint. Probing does
not disturb training: the idle gate only counts requests that can occupy the
GPUs — writes under `/v1/` plus a few engine endpoints — so `GET` traffic never
resets the idle timer and never pauses the trainer.

The `startupProbe` is the one that matters: `failureThreshold: 180` at
`periodSeconds: 10` gives startup 30 minutes for a cold weight download plus
engine warmup, and liveness/readiness only begin after it passes. If your model
is large or your registry is slow, raise it rather than fighting a
CrashLoopBackOff.

### Rollouts

At one replica a StatefulSet terminates the old pod before creating the new
one, which is exactly what you want: the GPU and the `ReadWriteOnce` volume are
released before the replacement is scheduled. A Deployment needs
`strategy: Recreate` to behave the same way — with the default `RollingUpdate`
the new pod waits forever for a GPU the old pod still holds.

Expect a full model reload on every rollout. There is no hot restart.

## Variants

### Fetch the plugin at pod start instead

If you would rather track a git ref than re-create a ConfigMap, swap the
`plugin-src` volume for an `emptyDir` and add an init container. Reuse the vLLM
image — it is already on the node, so this costs no extra pull:

```yaml
      initContainers:
        - name: fetch-plugin
          image: vllm/vllm-openai:latest        # same tag as the server below
          command: ["/bin/sh", "-c"]
          args:
            - >-
              mkdir -p /src/vllm_colocate &&
              curl -fsSL https://github.com/flozi00/vllm-self-spec/archive/refs/heads/main.tar.gz
              | tar -xz --strip-components=1 -C /src/vllm_colocate
          volumeMounts:
            - name: plugin-src
              mountPath: /src
      volumes:
        - name: plugin-src
          emptyDir: {}
```

Also drop `readOnly: true` from the server's `plugin-src` mount. Pin
`refs/heads/main` to `refs/tags/<tag>` or `archive/<sha>.tar.gz` so a pod
restart cannot pick up different code than its neighbours.

A tarball rather than `git clone` because the official image ships `curl` but
not `git`. If you prefer a real clone, use a dedicated git image
(`command: ["git"]`, `args: ["clone", "--depth=1", ...]`) — at the cost of a
second image your cluster must be allowed to pull. Either way this needs egress
to GitHub on every pod start.

### Use a purpose-built image

For GitOps, air-gapped clusters, or anywhere a pod must start with no
dependencies at all, build the repository's `Dockerfile` (the official image
plus three files), push it, and then delete from the StatefulSet:

- the `plugin-src` volume and its mount,
- the `PYTHONPATH` env var (the image sets it),
- the `command` override (the image's entrypoint is already
  `python -m vllm_colocate.app`).

Set `image:` to your build — or pin it centrally through the `images:` block in
`kustomization.yaml`.

### Run as non-root

The stock image runs as root but ships a `vllm` user (UID 2000, GID 0) with
group-writable caches, so restricted clusters can use:

```yaml
      securityContext:
        runAsUser: 2000
        runAsGroup: 0
        fsGroup: 0
```

`fsGroup` makes the two PVCs group-writable. Every path the process writes must
be one of those volumes — the shipped ConfigMap already points `HF_HOME`,
`VLLM_CACHE_ROOT` and `TRITON_CACHE_DIR` at them.

Namespaces that enforce the `restricted` Pod Security Standard also want
`runAsNonRoot: true`, `allowPrivilegeEscalation: false`,
`capabilities: {drop: ["ALL"]}` and `seccompProfile: {type: RuntimeDefault}` on
the container. Do not add `readOnlyRootFilesystem` — torch and the CUDA
toolchain write to the container filesystem at startup.

### Share the model cache

`hf-cache` is a per-pod volumeClaimTemplate. To share one cache across several
deployments, pre-create a `ReadWriteMany` PVC, delete that volumeClaimTemplate,
and mount the claim instead:

```yaml
      volumes:
        - name: hf-cache
          persistentVolumeClaim:
            claimName: shared-hf-cache
```

The checkpoint volume (`data`) must stay single-writer.

## Exposure and access control

**The training route is not authenticated.** vLLM's `--api-key` middleware
only guards paths under `/v1`, `/v2`, `/inference` and `/cohere`. `/train/sft`
is outside all of them, so an API key does nothing for it: anyone who can reach
port 8000 can queue training jobs that permanently change what your model says.

Practical consequences:

- Keep the Service `ClusterIP`. No `LoadBalancer`, no `NodePort`.
- If you publish it, route `/v1` only. The example in
  [`ingress.example.yaml`](../deploy/kubernetes/ingress.example.yaml) does
  exactly that, and `/train/sft` stays reachable only from inside the cluster.
- A NetworkPolicy can restrict *who* reaches the pod, but not *which paths* —
  both APIs are on one port. Use it as defense in depth, not as the boundary.
- Submitting training data from inside the cluster needs no ingress at all:

  ```shell
  kubectl -n vllm-colocate run trainer-submit --rm -i --restart=Never \
    --image=curlimages/curl:latest -- \
    curl -fsS -X POST http://vllm-colocate:8000/train/sft \
      -H 'content-type: application/json' \
      -d @- < samples.json
  ```

Three more things to fix at the proxy if you do expose `/v1`:

- **Streaming**: turn off response buffering, or token streaming arrives in one
  lump at the end.
- **Timeouts**: generations outlive a 60-second default read timeout, and
  `POST /train/sft?wait=true` can block for the length of a training job —
  prefer submitting without `wait` and polling `GET /train/sft`.
- **Body size**: training payloads are as large as the batch you submit. Nginx
  ingress caps bodies at 1 MB by default and answers 413; raise
  `proxy-body-size` for any client that submits real datasets.

## Day-2 operations

**Watch training.** `GET /train/sft` returns the queue, job history, the latest
checkpoint step, the active adapter, the idle-gate state and recent step
metrics:

```shell
kubectl -n vllm-colocate exec statefulset/vllm-colocate -- \
  curl -fsS localhost:8000/train/sft | jq '.training, .inference'
```

**Training never starts.** Check `inference.busy`. Steps only run once the
server has been idle for `VLLM_COLOCATE_IDLE_GRACE_SECONDS`, and continuous
production traffic can hold that off indefinitely. Probes and other `GET`
traffic do not count; `POST /v1/...` does. Either accept training as
off-peak work or give it a maintenance window.

**The data volume only grows.** Every submitted job is written to
`/data/vllm_colocate/jobs/<job-id>.jsonl` and kept for reproducibility —
nothing prunes it. Checkpoints self-prune to
`VLLM_COLOCATE_KEEP_LAST_CHECKPOINTS`, but the job files do not. Size the
volume for your submission rate, and clean up old job files periodically:

```shell
kubectl -n vllm-colocate exec statefulset/vllm-colocate -- \
  find /data/vllm_colocate/jobs -name '*.jsonl' -mtime +30 -delete
```

**Back up what was learned.** The adapters and checkpoints under
`/data/vllm_colocate/checkpoints/<model-slug>/` (for example
`Qwen--Qwen3-8B/`) are the only artifacts that cannot be rebuilt. Snapshot the
PVC, or copy the newest adapter out:

```shell
kubectl -n vllm-colocate exec statefulset/vllm-colocate -- \
  cat /data/vllm_colocate/checkpoints/Qwen--Qwen3-8B/latest.json

kubectl -n vllm-colocate cp \
  vllm-colocate-0:/data/vllm_colocate/checkpoints/Qwen--Qwen3-8B/adapter-step-00000128 \
  ./adapter-step-00000128
```

That directory is a plain PEFT adapter — it loads anywhere PEFT or vLLM LoRA
does.

**Node drains.** A drain interrupts training. Checkpoints are exported every
`VLLM_COLOCATE_CHECKPOINT_EVERY` steps, so an interrupted job loses at most the
steps since the last export and the next job resumes from that checkpoint. If
you would rather a drain wait for a human, apply
[`poddisruptionbudget.yaml`](../deploy/kubernetes/poddisruptionbudget.yaml) —
and know that at one replica it blocks every voluntary eviction.

**Do not set `VLLM_ALLOW_RUNTIME_LORA_UPDATING=0`.** The launcher enables it
before vLLM is imported; overriding it in the ConfigMap breaks the adapter hot
swap, and trained adapters will silently never reach the running server.

## Troubleshooting

| Symptom | Look at |
| --- | --- |
| Pod `Pending`, event `Insufficient nvidia.com/gpu` | No schedulable GPU node, or another pod holds the GPU (a previous replica that has not terminated) |
| `CrashLoopBackOff` with `No module named vllm_colocate` | The `vllm-colocate-src` ConfigMap is missing, or keys are not named `__init__.py` / `app.py` / `lora_trainer.py`, or `PYTHONPATH` is not `/opt` |
| Pod restarts before it ever serves | `startupProbe` too short for the weight download — raise `failureThreshold` |
| `OOMKilled` during a training job | Container memory limit; the trainer's RAM counts against it |
| CUDA OOM in the logs when a job starts | `VLLM_COLOCATE_GPU_MEMORY_UTILIZATION` too high, or `VLLM_COLOCATE_MAX_MODEL_LEN` unset on a long-context model |
| Server starts but `/train/sft` is 404 | The image entrypoint ran instead of the launcher — check the `command` override |
| Jobs stay `queued` | `inference.busy` in `GET /train/sft`; the gate is holding the trainer |
| Trained answers do not show up | Request the adapter model name (`<served-model-name>-lora`); check `training.active_adapter` and the job's `sync` result |
| 413 from the ingress on a training POST | Proxy body-size limit |
| Weights re-download on every restart | `HF_HOME` not pointing at the mounted cache volume |
