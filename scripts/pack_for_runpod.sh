#!/usr/bin/env bash
# Build a tarball you can scp to RunPod or upload via Jupyter.
# Excludes venv, large outputs, API keys, and git metadata.
# Unpacks to a single top-level folder (repo name).
# Usage: from repo root: bash scripts/pack_for_runpod.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT="$(dirname "$ROOT")"
NAME="$(basename "$ROOT")"
STAMP=$(date +%Y%m%d-%H%M)
ARCHIVE_NAME="miso-mini-runpod-${STAMP}.tar.gz"
ARCHIVE="$ROOT/$ARCHIVE_NAME"

cd "$PARENT"
# macOS: avoid xattr blobs that Linux tar warns about (LIBARCHIVE.xattr.com.apple.provenance)
if [[ "$(uname -s)" == "Darwin" ]]; then
  export COPYFILE_DISABLE=1
fi
tar czvf "$ARCHIVE" \
  --exclude="${NAME}/.venv" \
  --exclude="${NAME}/outputs" \
  --exclude="${NAME}/__pycache__" \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude="${NAME}/openai_key.txt" \
  --exclude="${NAME}/.git" \
  --exclude="${NAME}/local" \
  --exclude="${NAME}/*.tar.gz" \
  "$NAME"

echo ""
echo "Created: $ARCHIVE"
echo "On the pod: tar xzvf $ARCHIVE_NAME && cd $NAME"
