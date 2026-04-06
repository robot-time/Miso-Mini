# Build from repo root:
#   docker build -f serverless/Dockerfile -t your-dockerhub/miso-triad-serverless:latest .
#
# At run time, mount your triad LoRAs at TRIAD_ADAPTERS_DIR (default /app/adapters), or bake them in with an extra COPY.
#
# RunPod: attach a Network Volume with:
#   reasoning/  response/  critic/
# and set env TRIAD_ADAPTERS_DIR to that mount path.

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV TRIAD_ADAPTERS_DIR=/app/adapters
ENV TRIAD_CONFIG=/app/configs/triad_reasoning.yaml

WORKDIR /app

COPY serverless/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY configs/triad_reasoning.yaml /app/configs/triad_reasoning.yaml
COPY scripts/infer_triad.py /app/infer_triad.py
COPY serverless/handler.py /app/handler.py

RUN mkdir -p /app/adapters/reasoning /app/adapters/response /app/adapters/critic

# Optional: uncomment to bake adapters into the image (large).
# COPY outputs/triad_adapters/reasoning /app/adapters/reasoning
# COPY outputs/triad_adapters/response /app/adapters/response
# COPY outputs/triad_adapters/critic /app/adapters/critic

CMD ["python", "-u", "/app/handler.py"]
