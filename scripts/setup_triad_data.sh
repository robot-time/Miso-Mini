#!/usr/bin/env bash
# Prepare data/reasoning_sft.jsonl, response_sft.jsonl (WildChat + [adapter:response]), critic_sft.jsonl (Miso).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
MAX_ROWS="${WILDCHAT_MAX_ROWS:-8000}"

echo "== Triad data setup =="

"$PYTHON" scripts/bootstrap_reasoning_sft.py

gsm_lines=0
if [[ -f "data/reasoning_gsm8k.jsonl" ]]; then
  gsm_lines=$(wc -l < data/reasoning_gsm8k.jsonl | tr -d ' ')
fi

if [[ "$gsm_lines" -gt 0 ]]; then
  echo "Merging data/reasoning_seed.jsonl + data/reasoning_gsm8k.jsonl -> data/reasoning_sft.jsonl ($gsm_lines GSM8K rows)"
  "$PYTHON" scripts/merge_jsonl.py -o data/reasoning_sft.jsonl data/reasoning_seed.jsonl data/reasoning_gsm8k.jsonl
else
  reasoning_lines=0
  if [[ -f "data/reasoning_sft.jsonl" ]]; then
    reasoning_lines=$(wc -l < data/reasoning_sft.jsonl | tr -d ' ')
  fi
  if [[ "$reasoning_lines" -gt 0 ]]; then
    echo "Using existing data/reasoning_sft.jsonl ($reasoning_lines lines) — add data/reasoning_gsm8k.jsonl to merge seed+GSM8K automatically."
  else
    cp data/reasoning_seed.jsonl data/reasoning_sft.jsonl
    echo "Wrote data/reasoning_sft.jsonl from seed only. Stronger reasoning: python scripts/gsm8k_to_reasoning_sft.py --output data/reasoning_gsm8k.jsonl then re-run this script."
  fi
fi

lines=0
if [[ -f "data/response_sft.jsonl" ]]; then
  lines=$(wc -l < data/response_sft.jsonl | tr -d ' ')
fi
if [[ ! -f "data/response_sft.jsonl" ]] || [[ "$lines" -lt 1 ]]; then
  echo "Building response_sft from WildChat (tagged [adapter:response])..."
  "$PYTHON" scripts/wildchat_to_sft.py \
    --parquet-only \
    --max-rows "$MAX_ROWS" \
    --output data/response_sft.jsonl \
    --system-prefix "[adapter:response]"
else
  echo "Using existing data/response_sft.jsonl"
fi

if [[ ! -f "data/miso_raw.jsonl" ]]; then
  echo "Need data/miso_raw.jsonl for critic adapter."
  exit 1
fi
"$PYTHON" scripts/convert_to_sft.py \
  --input data/miso_raw.jsonl \
  --output data/critic_sft.jsonl \
  --system-prefix "[adapter:critic]"
echo "Built data/critic_sft.jsonl from miso_raw"

echo "Done. Files:"
echo "  data/reasoning_sft.jsonl   — CoT / step-by-step (expand this!)"
echo "  data/response_sft.jsonl    — conversational (WildChat + response tag)"
echo "  data/critic_sft.jsonl      — critique / refine (Miso + critic tag)"
