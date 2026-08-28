ARG VLLM_BASE_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_BASE_IMAGE}

WORKDIR /opt

# torch/transformers/safetensors come from the vLLM base image; the
# upgrade strategy keeps pip from clobbering its CUDA-matched torch build.
RUN python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
    "fastapi>=0.115.0" \
    "httpx>=0.28.0" \
    "huggingface_hub>=0.20.0" \
    "safetensors>=0.4.0"

RUN mkdir -p /opt/vllm_colocate

COPY __init__.py /opt/vllm_colocate/
COPY app.py /opt/vllm_colocate/
COPY lora_trainer.py /opt/vllm_colocate/

ENV PYTHONPATH=/opt

# 8000: vLLM OpenAI-compatible inference + the /train/sft route, one port
EXPOSE 8000

ENTRYPOINT ["python", "-m", "vllm_colocate.app"]
