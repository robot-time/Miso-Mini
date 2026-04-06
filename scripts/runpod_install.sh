#!/usr/bin/env bash
# One-shot environment setup for a Linux CUDA machine (e.g. RunPod).
# From repo root: bash scripts/runpod_install.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if ! command -v "$PY" &>/dev/null; then
  echo "Need python3 on PATH."
  exit 1
fi

echo "== Creating venv at $ROOT/.venv =="
"$PY" -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install -U pip wheel setuptools
python -m pip install -r requirements.txt

echo ""
echo "== Verifying PyTorch + CUDA =="
python <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo ""
echo "Next (triad training on GPU):"
echo "  source .venv/bin/activate"
echo "  export FORCE_TRAINER=unsloth   # use Unsloth on CUDA"
echo "  ./scripts/setup_triad_data.sh   # needs data/miso_raw.jsonl; WildChat step downloads data"
echo "  ./scripts/train_triad_adapters.sh"
echo ""
echo "If Hugging Face downloads fail, run: huggingface-cli login"
