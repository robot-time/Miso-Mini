#!/usr/bin/env bash
# Poll RunPod until benchmark_suite.py finishes, then SCP results to local eval/.
#
#   SSH_PORT=40140 SSH_HOST=root@213.192.2.118 SSH_KEY=~/.ssh/id_ed25519 \
#     bash scripts/monitor_runpod_benchmark.sh
#
# Logs to eval/monitor_runpod.log (append).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${ROOT}/eval/monitor_runpod.log"
REMOTE_DIR="/workspace/Miso-Mini/eval/benchmark_run_full"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_HOST="${SSH_HOST:-root@213.192.2.118}"
SSH_PORT="${SSH_PORT:-40140}"
INTERVAL_SEC="${INTERVAL_SEC:-600}"
MAX_ITER="${MAX_ITER:-400}"

ssh_cmd() {
  ssh -o ConnectTimeout=30 -o BatchMode=yes -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" "$@"
}

mkdir -p "$ROOT/eval"
{
  echo "=== monitor start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "host=$SSH_HOST port=$SSH_PORT interval=${INTERVAL_SEC}s max_iter=$MAX_ITER"
} | tee -a "$LOG"

for i in $(seq 1 "$MAX_ITER"); do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if ! out="$(ssh_cmd "pgrep -af 'scripts/benchmark_suite.py' 2>/dev/null || true; pgrep -af 'python -m lm_eval run' 2>/dev/null || true" 2>&1)"; then
    echo "$ts SSH_FAIL $out" | tee -a "$LOG"
    sleep "$INTERVAL_SEC"
    continue
  fi

  if echo "$out" | grep -qE 'benchmark_suite\.py|lm_eval run'; then
    echo "$ts iter=$i RUNNING" | tee -a "$LOG"
    # One-line progress from remote log
    prog="$(ssh_cmd "tail -c 500 /workspace/Miso-Mini/eval/benchmark_suite.log 2>/dev/null | tr '\r' '\n' | tail -1" 2>/dev/null || echo "?")"
    echo "  progress: ${prog:0:200}" >> "$LOG"
  else
    echo "$ts iter=$i FINISHED (no benchmark process)" | tee -a "$LOG"
    ssh_cmd "ls -la $REMOTE_DIR 2>/dev/null; test -f /workspace/Miso-Mini/eval/benchmark_run_full/manifest.json && cat /workspace/Miso-Mini/eval/benchmark_run_full/manifest.json" >> "$LOG" 2>&1 || true

    echo "Pulling results to $ROOT/eval/runpod_benchmark_run_full ..." | tee -a "$LOG"
    rm -rf "${ROOT}/eval/runpod_benchmark_run_full"
    set +e
    scp -o ConnectTimeout=120 -r -P "$SSH_PORT" -i "$SSH_KEY" \
      "${SSH_HOST}:${REMOTE_DIR}" "${ROOT}/eval/runpod_benchmark_run_full" >> "$LOG" 2>&1
    scp_rc=$?
    set -e
    if [[ "$scp_rc" -ne 0 ]]; then
      echo "WARNING: scp benchmark_run_full failed (rc=$scp_rc); check SSH and remote path." | tee -a "$LOG"
    fi

    set +e
    scp -o ConnectTimeout=60 -P "$SSH_PORT" -i "$SSH_KEY" \
      "${SSH_HOST}:/workspace/Miso-Mini/eval/benchmark_suite.log" \
      "${ROOT}/eval/runpod_benchmark_suite_final.log" >> "$LOG" 2>&1
    set -e

    echo "=== monitor done OK $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
    exit 0
  fi

  sleep "$INTERVAL_SEC"
done

echo "=== monitor gave up after $MAX_ITER iterations $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
exit 1
