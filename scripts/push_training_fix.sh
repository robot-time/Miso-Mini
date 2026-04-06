#!/usr/bin/env bash
# Push fixed train_unsloth.py + train_lora_trl.py to your GPU host (run on your Mac).
# Edit HOST/PORT/KEY if needed, or set REMOTE_ROOT.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${DEPLOY_HOST:-213.192.2.118}"
PORT="${DEPLOY_PORT:-40060}"
KEY="${DEPLOY_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE="${REMOTE_ROOT:-root@${HOST}:/workspace/Miso-Mini/scripts/}"

echo "Pushing training scripts -> $REMOTE"
scp -P "$PORT" -i "$KEY" -o IdentitiesOnly=yes \
  "$ROOT/scripts/train_unsloth.py" \
  "$ROOT/scripts/train_lora_trl.py" \
  "$REMOTE"
echo "Done. On the server: cd /workspace/Miso-Mini && source .venv/bin/activate && ./scripts/train_triad_adapters.sh"
