#!/usr/bin/env bash
# One-shot SFT on a 70/30 (default) mix of WildChat + Miso SFT JSONL.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
RATIO="${WILDCHAT_RATIO:-0.7}"
TARGET="${MIX_TARGET_SIZE:-8000}"
OUT_MIX="${MIX_OUT:-data/mixed_sft.jsonl}"
TRAIN_OUT="${TRAIN_OUT:-outputs/phi4-mixed-lora}"

WC_SFT="${WC_SFT:-data/wildchat_sft.jsonl}"
MISO_SFT="${MISO_SFT:-data/miso_sft.jsonl}"

TAG_FLAG=()
if [[ "${TAG_STYLE:-0}" == "1" ]]; then
  TAG_FLAG=(--tag-style)
fi

if [[ ! -f "$WC_SFT" ]] || [[ ! -f "$MISO_SFT" ]]; then
  echo "Need $WC_SFT and $MISO_SFT (build with wildchat_to_sft.py and convert_to_sft.py)"
  exit 1
fi

echo "Merging (wildchat_ratio=$RATIO, target_size=$TARGET) ..."
"$PYTHON" scripts/merge_sft_mix.py \
  --wildchat "$WC_SFT" \
  --miso "$MISO_SFT" \
  --wildchat-ratio "$RATIO" \
  --target-size "$TARGET" \
  --output "$OUT_MIX" \
  "${TAG_FLAG[@]}"

if [[ "${FORCE_TRAINER:-}" == "hf" ]]; then
  TRAIN_SCRIPT="$ROOT/scripts/train_lora_trl.py"
elif [[ "${FORCE_TRAINER:-}" == "unsloth" ]]; then
  TRAIN_SCRIPT="$ROOT/scripts/train_unsloth.py"
elif "$PYTHON" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  TRAIN_SCRIPT="$ROOT/scripts/train_unsloth.py"
else
  TRAIN_SCRIPT="$ROOT/scripts/train_lora_trl.py"
fi

echo "Training on $OUT_MIX -> $TRAIN_OUT ($TRAIN_SCRIPT)"
"$PYTHON" "$TRAIN_SCRIPT" \
  --config configs/train_phi4mini_qlora.yaml \
  --train-file "$OUT_MIX" \
  --output-dir "$TRAIN_OUT"

echo "Done -> $TRAIN_OUT"
