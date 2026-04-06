# RunPod Serverless / Docker Hub — triad worker (reasoning → response → critic).
# Build from repository root:
#   docker build -t miso-triad-serverless:latest .
#
# RunPod GitHub build: use Dockerfile path `Dockerfile` and context/repo root `.`
# (not `serverless/` as context — paths below assume root).

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV TRIAD_ADAPTERS_DIR=/app/adapters
ENV TRIAD_CONFIG=/app/configs/triad_reasoning.yaml
# Prefer PyTorch CUDA wheels if a dependency tries to pull torch from PyPI
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124
# XET downloads can need a lot of temp space; small serverless container disks → ENOSPC during hub downloads.
ENV HF_HUB_DISABLE_XET=1

WORKDIR /app

COPY serverless/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && python -c "import torch; import peft; import transformers; import runpod; import bitsandbytes; print('imports_ok', torch.__version__)"

# Serverless workers often have a small *writable* disk. Downloading Phi-4 at runtime fills it → errno 28.
# Default: bake `microsoft/Phi-4-mini-instruct` into the image at build time (~15GB+ larger image, no runtime download).
# Slim build (you provide HF_HOME on a network volume): docker build --build-arg EMBED_PHI4_BASE=0 ...
ARG EMBED_PHI4_BASE=1
RUN if [ "$EMBED_PHI4_BASE" = "1" ]; then \
      mkdir -p /app/hf_cache \
      && python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/Phi-4-mini-instruct', cache_dir='/app/hf_cache')" ; \
    fi

COPY configs/triad_reasoning.yaml /app/configs/triad_reasoning.yaml
COPY scripts/infer_triad.py /app/infer_triad.py
COPY serverless/handler.py /app/handler.py

RUN mkdir -p /app/adapters/reasoning /app/adapters/response /app/adapters/critic

# Optional: bake adapters (uncomment)
# COPY outputs/triad_adapters/reasoning /app/adapters/reasoning
# COPY outputs/triad_adapters/response /app/adapters/response
# COPY outputs/triad_adapters/critic /app/adapters/critic

CMD ["python", "-u", "/app/handler.py"]
