#!/usr/bin/env bash
# Cloud GPU (e.g. RunPod): run setup + triad training, then halt the machine
# only if every step exits 0. Uses the shell chain — not embedded in Python —
# so your MacBook never runs shutdown by accident.
#
# Usage:
#   source .venv/bin/activate
#   export FORCE_TRAINER=unsloth
#   ./scripts/run_triad_on_runpod.sh
#
# Dry run (train but do not halt):
#   SKIP_SHUTDOWN=1 ./scripts/run_triad_on_runpod.sh
#
# Data already built — skip setup_triad_data.sh:
#   RUN_SETUP=0 ./scripts/run_triad_on_runpod.sh
#
# Optional grace period before halt (seconds):
#   HALT_DELAY_SEC=120 ./scripts/run_triad_on_runpod.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${RUN_SETUP:-1}" == "1" ]]; then
  ./scripts/setup_triad_data.sh
fi

./scripts/train_triad_adapters.sh

echo ""
echo "=== Triad training finished successfully ==="

if [[ "${SKIP_SHUTDOWN:-}" == "1" ]]; then
  echo "SKIP_SHUTDOWN=1 — not halting. Stop the pod manually when you are done."
  exit 0
fi

if [[ "${HALT_DELAY_SEC:-0}" =~ ^[0-9]+$ ]] && [[ "${HALT_DELAY_SEC:-0}" -gt 0 ]]; then
  echo "Halting host in ${HALT_DELAY_SEC}s (Ctrl+C aborts this script; download adapters first if needed)..."
  sleep "$HALT_DELAY_SEC"
fi

if command -v shutdown >/dev/null 2>&1; then
  shutdown -h now 2>/dev/null || sudo shutdown -h now
else
  echo "No 'shutdown' in this environment — use the RunPod dashboard to stop the pod."
fi
