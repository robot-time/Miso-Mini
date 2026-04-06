#!/usr/bin/env bash
# Apply a pack_for_runpod.sh tarball over an existing Miso-Mini checkout without rsync.
# Keeps .venv and outputs/ in the target directory.
#
# Usage on the pod:
#   bash scripts/apply_runpod_bundle.sh /workspace/miso-mini-runpod-20260403-1751.tar.gz
#   bash scripts/apply_runpod_bundle.sh /path/to/archive.tar.gz /workspace/Miso-Mini
set -euo pipefail

ARCHIVE="${1:?Usage: $0 /path/to/miso-mini-runpod-*.tar.gz [target_dir]}"
TARGET="${2:-/workspace/Miso-Mini}"

if [[ ! -f "$ARCHIVE" ]]; then
  echo "File not found: $ARCHIVE"
  echo "Tip: ls /workspace/*.tar.gz  and use the full path (not / or ~ unless the file is there)."
  exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

tar xzf "$ARCHIVE" -C "$TMP"
SRC="$TMP/Miso-Mini"
if [[ ! -d "$SRC" ]]; then
  echo "Expected a top-level Miso-Mini/ folder in the archive; got:"
  ls -la "$TMP"
  exit 1
fi

mkdir -p "$TARGET"
while IFS= read -r -d '' item; do
  name=$(basename "$item")
  [[ "$name" == ".venv" ]] && continue
  [[ "$name" == "outputs" ]] && continue
  rm -rf "$TARGET/$name"
  cp -a "$item" "$TARGET/"
done < <(find "$SRC" -mindepth 1 -maxdepth 1 -print0)

echo "Updated $TARGET (preserved .venv and outputs if present)."
