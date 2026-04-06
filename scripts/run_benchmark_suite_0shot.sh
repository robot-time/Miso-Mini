#!/usr/bin/env bash
# Wrapper: legacy single lm-eval run — gsm8k + mmlu + arc_challenge, all 0-shot.
# For the main matrix (GSM8K 0/8-shot, HellaSwag 5, ARC 10, BBH-CoT): run_benchmark_suite.sh
#
#   bash scripts/run_benchmark_suite_0shot.sh [MODEL_DIR]
#   bash scripts/run_benchmark_suite_0shot.sh outputs/merged_reasoning --limit 5
#
# Extra args are passed to the Python script (forwards to benchmark_suite.py --legacy).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$ROOT/scripts/benchmark_suite_0shot.py" "$@"
