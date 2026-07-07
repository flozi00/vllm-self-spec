ARG VLLM_BASE_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_BASE_IMAGE}

WORKDIR /opt

RUN python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
    "fastapi>=0.115.0" \
    "httpx>=0.28.0" \
    "uvicorn>=0.34.0" \
    "huggingface_hub>=0.20.0" \
    arctic-inference

RUN mkdir -p /opt/vllm_dflash_jit

COPY __init__.py /opt/vllm_dflash_jit/
COPY app.py /opt/vllm_dflash_jit/
COPY hub_checkpoints.py /opt/vllm_dflash_jit/
COPY hybrid_proposer.py /opt/vllm_dflash_jit/
COPY jetspec_adapter.py /opt/vllm_dflash_jit/
COPY live_lora_trainer.py /opt/vllm_dflash_jit/
COPY paths.py /opt/vllm_dflash_jit/
COPY sitecustomize.py /opt/vllm_dflash_jit/
COPY suffix_cache.py /opt/vllm_dflash_jit/
COPY trainer.py /opt/vllm_dflash_jit/
COPY templates /opt/vllm_dflash_jit/templates
COPY _vllm_dflash_jit_rust.so /opt/vllm_dflash_jit/
COPY sitecustomize.py /opt/sitecustomize.py

ENV PYTHONPATH=/opt
EXPOSE 30006

ENTRYPOINT ["python", "-m", "vllm_dflash_jit.app"]
