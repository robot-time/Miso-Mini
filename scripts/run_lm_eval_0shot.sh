#!/usr/bin/env bash
# Run lm-evaluation-harness with ONE consistent setting: --num_fewshot 0
#
# Usage:
#   bash scripts/run_lm_eval_0shot.sh [MODEL_DIR]
#
# Env:
#   MODEL_DIR   default: outputs/merged_reasoning
#   TASKS       comma-separated (default: gsm8k,mmlu,arc_challenge)
#               Add hellaswag if desired. mmlu is slow on one GPU.
#   OUTPUT_PATH path to results .json (default: eval/lm_eval_0shot_<timestamp>.json)
#   DEVICE      default: cuda
#   LM_EVAL_LIMIT  if set, passed as --limit (smoke test only)
#
# Example (core three, no MMLU):
#   bash scripts/run_lm_eval_0shot.sh outputs/merged_reasoning
#
# Example (full four tasks including MMLU):
#   TASKS=gsm8k,mmlu,arc_challenge,hellaswag OUTPUT_PATH=eval/miso_0shot.json \
#     bash scripts/run_lm_eval_0shot.sh outputs/merged_reasoning

set -euo pipefail

MODEL_DIR="${1:-${MODEL_DIR:-outputs/merged_reasoning}}"
TASKS="${TASKS:-gsm8k,mmlu,arc_challenge}"
OUT="${OUTPUT_PATH:-eval/lm_eval_0shot_$(date +%Y%m%d_%H%M%S).json}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  echo "Merge an adapter first, e.g.:" >&2
  echo "  python scripts/merge_adapter_for_eval.py --adapter outputs/triad_adapters/reasoning --output outputs/merged_reasoning" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

echo "model:     $MODEL_DIR"
echo "tasks:     $TASKS"
echo "fewshot:   0"
echo "output:    $OUT"
if [[ -n "${LM_EVAL_LIMIT:-}" ]]; then
  echo "limit:     ${LM_EVAL_LIMIT} (smoke test)"
fi
echo

EXTRA=()
if [[ -n "${LM_EVAL_LIMIT:-}" ]]; then
  EXTRA=(--limit "${LM_EVAL_LIMIT}")
fi

lm_eval run \
  --model hf \
  --model_args "pretrained=${MODEL_DIR}" \
  --tasks "${TASKS}" \
  --num_fewshot 0 \
  --device "${DEVICE}" \
  --batch_size auto \
  --output_path "${OUT}" \
  "${EXTRA[@]}"

echo
echo "Wrote ${OUT}"
