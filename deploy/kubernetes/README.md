# Kubernetes manifests

Ready-to-apply manifests that run this plugin on the **official**
`vllm/vllm-openai` image — no custom image, no build step.

Full walkthrough, variants and operational notes:
[`docs/kubernetes.md`](../../docs/kubernetes.md).

```shell
# from the repository root
kubectl apply -f deploy/kubernetes/namespace.yaml

kubectl -n vllm-colocate create configmap vllm-colocate-src \
  --from-file=__init__.py --from-file=app.py --from-file=lora_trainer.py

kubectl apply -k deploy/kubernetes/
```

| File | Purpose |
| --- | --- |
| `namespace.yaml` | `vllm-colocate` namespace |
| `configmap.yaml` | Launcher tunables (model, memory split, idle grace, cache paths) |
| `statefulset.yaml` | The single server pod: stock vLLM image + plugin source + volumes |
| `service.yaml` | ClusterIP for clients, headless service for stable pod DNS |
| `secret.example.yaml` | Shape of the optional `HF_TOKEN` secret (placeholder, not applied) |
| `poddisruptionbudget.yaml` | Optional: make node drains wait for a running job |
| `ingress.example.yaml` | Optional: expose `/v1` only, never `/train/sft` |

`kustomization.yaml` wires up the first four. The rest are opt-in — uncomment
them there once you have read what they do.
