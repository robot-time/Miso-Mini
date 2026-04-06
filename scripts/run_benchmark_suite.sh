#!/usr/bin/env bash
# Wrapper: benchmark_suite.py — mixed-shot matrix (GSM8K 0-shot, HellaSwag, ARC, MMLU-mini, BBH-CoT).
#
#   bash scripts/run_benchmark_suite.sh [MODEL_DIR]
#   bash scripts/run_benchmark_suite.sh outputs/merged_reasoning --limit 5
#
# Old 0-shot trio (gsm8k, mmlu, arc_challenge): benchmark_suite_0shot.sh or benchmark_suite.py --legacy

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$ROOT/scripts/benchmark_suite.py" "$@"
