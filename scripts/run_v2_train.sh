#!/usr/bin/env bash
# V2 balanced LoRA: build mix (40/30/30) then train from BASE ONLY — do not pass --adapter-dir.
#
#   bash scripts/run_v2_train.sh
#   TOTAL_ROWS=8000 OUTPUT_ADAPTER=outputs/lora_v2_balanced bash scripts/run_v2_train.sh
#
# Requires CUDA + Unsloth (see requirements.txt). Run from repo root.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MIX_OUT="${MIX_OUT:-data/v2_mix_sft.jsonl}"
TOTAL_ROWS="${TOTAL_ROWS:-12000}"
OUTPUT_ADAPTER="${OUTPUT_ADAPTER:-outputs/lora_v2_balanced}"
CONFIG="${CONFIG:-configs/train_v2_balanced.yaml}"

echo "==> Building mix -> $MIX_OUT ($TOTAL_ROWS rows)"
python3 scripts/build_v2_mix.py --output "$MIX_OUT" --total-rows "$TOTAL_ROWS"

echo "==> Training fresh LoRA -> $OUTPUT_ADAPTER (no --adapter-dir)"
python3 scripts/train_unsloth.py \
  --config "$CONFIG" \
  --train-file "$MIX_OUT" \
  --output-dir "$OUTPUT_ADAPTER"

echo "Done. Merge for eval, e.g.:"
echo "  python scripts/merge_adapter_for_eval.py --adapter $OUTPUT_ADAPTER --output outputs/merged_v2"
