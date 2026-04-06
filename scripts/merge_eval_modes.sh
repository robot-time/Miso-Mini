#!/usr/bin/env bash
# Merge base + one LoRA for lm-eval (each mode = one merged folder).
#
# Usage:
#   bash scripts/merge_eval_modes.sh
#
# Writes:
#   outputs/merged_response   ← triad_adapters/response (baseline: answer without triad chain)
#   outputs/merged_reasoning  ← triad_adapters/reasoning (math/reasoning head)
#
# Full triad (reasoning → response → critic) is NOT a single HF merge; benchmark it with
# scripts/eval_triad_task.py (see docs/BENCHMARK.md).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE="${BASE:-microsoft/Phi-4-mini-instruct}"
AD="${AD:-outputs/triad_adapters}"

for pair in "response:merged_response" "reasoning:merged_reasoning"; do
  adapter="${pair%%:*}"
  out="${pair##*:}"
  echo "=== Merging ${AD}/${adapter} -> outputs/${out} ==="
  python scripts/merge_adapter_for_eval.py \
    --base "$BASE" \
    --adapter "${AD}/${adapter}" \
    --output "outputs/${out}"
done

echo
echo "Done. Run 0-shot suite on each merged dir, e.g.:"
echo "  TASKS=gsm8k,arc_challenge,hellaswag bash scripts/run_lm_eval_0shot.sh outputs/merged_response"
echo "  TASKS=gsm8k,arc_challenge,hellaswag bash scripts/run_lm_eval_0shot.sh outputs/merged_reasoning"
