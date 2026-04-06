#!/usr/bin/env bash
# Pack repo + triad adapters (+ optional configs) for uploading to a GPU host to run 0-shot lm-eval.
#
# Produces: miso-mini-gpu-eval-<timestamp>.tar.gz in the repo root.
#
# Includes:
#   scripts/, configs/, schemas/, prompts/, docs/BENCHMARK.md, requirements*.txt,
#   data/benchmark_results_template.json,
#   outputs/triad_adapters/   (your LoRAs — required for merge + eval)
#
# Excludes:
#   .venv, .git, openai_key.txt, large data/, runpod_eval snapshots, other outputs, __pycache__
#
# On the GPU:
#   tar xzvf miso-mini-gpu-eval-*.tar.gz --no-same-owner && cd Miso-Mini   # --no-same-owner avoids uid/gid errors from macOS archives
#   pip install -r requirements-eval.txt
#   python scripts/merge_adapter_for_eval.py --adapter outputs/triad_adapters/reasoning --output outputs/merged_reasoning
#   python scripts/benchmark_suite.py outputs/merged_reasoning --output-dir eval/benchmark_run
#   # download eval/results.json, then delete merged dir + HF cache if you need space
#
# Usage: from repo root —  bash scripts/pack_for_gpu_eval.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT="$(dirname "$ROOT")"
NAME="$(basename "$ROOT")"
STAMP=$(date +%Y%m%d-%H%M)
ARCHIVE_NAME="miso-mini-gpu-eval-${STAMP}.tar.gz"
ARCHIVE="$ROOT/$ARCHIVE_NAME"

cd "$PARENT"
if [[ "$(uname -s)" == "Darwin" ]]; then
  export COPYFILE_DISABLE=1
fi

tar czvf "$ARCHIVE" \
  --exclude="${NAME}/.venv" \
  --exclude="${NAME}/__pycache__" \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude="${NAME}/openai_key.txt" \
  --exclude="${NAME}/.git" \
  --exclude="${NAME}/local" \
  --exclude="${NAME}/runpod_backup" \
  --exclude="${NAME}/data/miso_raw.jsonl" \
  --exclude="${NAME}/data/*.jsonl" \
  --exclude="${NAME}/data/runpod_eval" \
  --exclude="${NAME}/outputs/phi4-stage2-miso" \
  --exclude="${NAME}/outputs/triad_adapters_mlx" \
  --exclude="${NAME}/outputs/figures" \
  --exclude="${NAME}/eval" \
  --exclude="${NAME}/*.tar.gz" \
  "${NAME}/scripts" \
  "${NAME}/configs" \
  "${NAME}/schemas" \
  "${NAME}/prompts" \
  "${NAME}/docs" \
  "${NAME}/requirements.txt" \
  "${NAME}/requirements-eval.txt" \
  "${NAME}/data/benchmark_results_template.json" \
  "${NAME}/outputs/triad_adapters"

echo ""
echo "Created: $ARCHIVE"
echo "Upload to GPU, unpack (use --no-same-owner if tar fails on ownership), merge adapters, run: python scripts/benchmark_suite.py outputs/merged_reasoning --output-dir eval/benchmark_run"
