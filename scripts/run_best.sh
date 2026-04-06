#!/usr/bin/env bash
# Recommended path: build WildChat SFT + Miso SFT (if needed), then curriculum train.
# Requires: GPU, venv with requirements.txt, Hugging Face access for WildChat.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
WILDCHAT_MAX_ROWS="${WILDCHAT_MAX_ROWS:-8000}"
WC_SFT="${WC_SFT:-data/wildchat_sft.jsonl}"
MISO_RAW="${MISO_RAW:-data/miso_raw.jsonl}"
MISO_SFT="${MISO_SFT:-data/miso_sft.jsonl}"

echo "== Miso Mini: curriculum (WildChat → Miso) =="
echo ""

lines=0
if [[ -f "$WC_SFT" ]]; then
  lines=$(wc -l < "$WC_SFT" | tr -d ' ')
fi
if [[ ! -f "$WC_SFT" ]] || [[ "$lines" -lt 1 ]]; then
  echo "[1/3] Building WildChat subset ($WILDCHAT_MAX_ROWS rows, no API cost)..."
  # --parquet-only avoids datasets.load_dataset (broken on some Python 3.14 + datasets builds).
  "$PYTHON" scripts/wildchat_to_sft.py \
    --parquet-only \
    --max-rows "$WILDCHAT_MAX_ROWS" \
    --output "$WC_SFT"
else
  echo "[1/3] Using existing $WC_SFT"
fi

if [[ ! -f "$MISO_SFT" ]]; then
  if [[ ! -f "$MISO_RAW" ]]; then
    echo "Missing $MISO_RAW — generate teacher data first (scripts/generate_synthetic.py + convert_to_sft)."
    exit 1
  fi
  echo "[2/3] Converting $MISO_RAW -> $MISO_SFT"
  "$PYTHON" scripts/convert_to_sft.py \
    --input "$MISO_RAW" \
    --output "$MISO_SFT" \
    --mode train
else
  echo "[2/3] Using existing $MISO_SFT"
fi

echo "[3/3] Training (stage 1 WildChat, stage 2 Miso LoRA on top)..."
exec "$ROOT/scripts/run_curriculum_train.sh"
