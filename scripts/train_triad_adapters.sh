#!/usr/bin/env bash
# One base model, three independent LoRA adapters (reasoning / response / critic).
# Run: ./scripts/setup_triad_data.sh && ./scripts/train_triad_adapters.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"

OUT="${TRIAD_OUT:-outputs/triad_adapters}"

if [[ "${FORCE_TRAINER:-}" == "hf" ]]; then
  TRAIN_SCRIPT="$ROOT/scripts/train_lora_trl.py"
elif [[ "${FORCE_TRAINER:-}" == "unsloth" ]]; then
  TRAIN_SCRIPT="$ROOT/scripts/train_unsloth.py"
elif "$PYTHON" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  TRAIN_SCRIPT="$ROOT/scripts/train_unsloth.py"
else
  TRAIN_SCRIPT="$ROOT/scripts/train_lora_trl.py"
fi

for f in data/reasoning_sft.jsonl data/response_sft.jsonl data/critic_sft.jsonl; do
  if [[ ! -f "$f" ]]; then
    echo "Missing $f — run ./scripts/setup_triad_data.sh first."
    exit 1
  fi
done

echo "Trainer: $TRAIN_SCRIPT"
mkdir -p "$OUT"

echo "=== [1/3] reasoning -> $OUT/reasoning ==="
"$PYTHON" "$TRAIN_SCRIPT" \
  --config configs/triad_reasoning.yaml \
  --train-file data/reasoning_sft.jsonl \
  --output-dir "$OUT/reasoning"

echo "=== [2/3] response -> $OUT/response ==="
"$PYTHON" "$TRAIN_SCRIPT" \
  --config configs/triad_response.yaml \
  --train-file data/response_sft.jsonl \
  --output-dir "$OUT/response"

echo "=== [3/3] critic -> $OUT/critic ==="
"$PYTHON" "$TRAIN_SCRIPT" \
  --config configs/triad_critic.yaml \
  --train-file data/critic_sft.jsonl \
  --output-dir "$OUT/critic"

echo ""
echo "Done. Adapters:"
echo "  $OUT/reasoning"
echo "  $OUT/response"
echo "  $OUT/critic"
echo "Inference: python scripts/infer_triad.py --adapters-dir $OUT --question \"...\""
