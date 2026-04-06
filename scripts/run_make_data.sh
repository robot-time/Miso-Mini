#!/usr/bin/env bash
# Generate synthetic JSONL (budgeted) + convert to SFT. Run from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
BUDGET="${BUDGET_USD:-3.5}"
MODEL="${TEACHER_MODEL:-gpt-5.4-mini}"
MAX_OUT="${MAX_OUTPUT_TOKENS:-650}"

if [[ ! -f "data/seeds_bulk.jsonl" ]]; then
  echo "Building data/seeds_bulk.jsonl ..."
  "$PYTHON" scripts/bootstrap_seeds.py
fi

APPEND=()
if [[ -f "data/miso_raw.jsonl" ]]; then
  APPEND=(--append)
  echo "Appending to existing data/miso_raw.jsonl"
fi

echo "Generating (budget this run: \$$BUDGET) ..."
"$PYTHON" scripts/generate_synthetic.py \
  --seed-file data/seeds_bulk.jsonl \
  --out-file data/miso_raw.jsonl \
  --teacher-model "$MODEL" \
  --budget-usd "$BUDGET" \
  --max-output-tokens "$MAX_OUT" \
  --samples-per-seed 1 \
  --shuffle-seeds \
  "${APPEND[@]}"

echo "Converting to SFT ..."
"$PYTHON" scripts/convert_to_sft.py \
  --input data/miso_raw.jsonl \
  --output data/miso_sft.jsonl \
  --mode train

N="$(wc -l < data/miso_raw.jsonl | tr -d ' ')"
echo ""
echo "Done. Raw samples: $N  |  SFT: data/miso_sft.jsonl"
echo "Re-run this script with more BUDGET_USD to grow the dataset (uses --append)."
echo "Train: $PYTHON scripts/train_unsloth.py --config configs/train_phi4mini_qlora.yaml --train-file data/miso_sft.jsonl --output-dir outputs/phi4mini-miso-lora"
