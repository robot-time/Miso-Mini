#!/usr/bin/env bash
# Two-stage SFT: (1) WildChat realism (2) Miso self-refine on top of stage-1 LoRA.
# Uses Unsloth on CUDA; falls back to train_lora_trl.py on Apple Silicon / CPU (Unsloth is GPU-vendor specific).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
STAGE1_OUT="${STAGE1_OUT:-outputs/phi4-stage1-wildchat}"
STAGE2_OUT="${STAGE2_OUT:-outputs/phi4-stage2-miso}"
WC_SFT="${WC_SFT:-data/wildchat_sft.jsonl}"
MISO_SFT="${MISO_SFT:-data/miso_sft.jsonl}"

if [[ ! -f "$WC_SFT" ]]; then
  echo "Missing $WC_SFT — run: $PYTHON scripts/wildchat_to_sft.py --parquet-only --max-rows 10000 --output $WC_SFT"
  exit 1
fi
if [[ ! -f "$MISO_SFT" ]]; then
  echo "Missing $MISO_SFT — run: $PYTHON scripts/convert_to_sft.py --input data/miso_raw.jsonl --output $MISO_SFT --mode train"
  exit 1
fi

if [[ "${FORCE_TRAINER:-}" == "hf" ]]; then
  TRAIN_SCRIPT="$ROOT/scripts/train_lora_trl.py"
  echo "Using Hugging Face + PEFT (FORCE_TRAINER=hf)."
elif [[ "${FORCE_TRAINER:-}" == "unsloth" ]]; then
  TRAIN_SCRIPT="$ROOT/scripts/train_unsloth.py"
  echo "Using Unsloth (FORCE_TRAINER=unsloth)."
elif "$PYTHON" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  TRAIN_SCRIPT="$ROOT/scripts/train_unsloth.py"
  echo "Using Unsloth (CUDA detected)."
else
  TRAIN_SCRIPT="$ROOT/scripts/train_lora_trl.py"
  echo "Using Hugging Face + PEFT — no CUDA (Unsloth requires NVIDIA/AMD/Intel GPU)."
fi

echo "=== Stage 1: WildChat ($WC_SFT) -> $STAGE1_OUT ==="
"$PYTHON" "$TRAIN_SCRIPT" \
  --config configs/train_stage1_wildchat.yaml \
  --train-file "$WC_SFT" \
  --output-dir "$STAGE1_OUT"

echo ""
echo "=== Stage 2: Miso refine ($MISO_SFT) -> $STAGE2_OUT (resume LoRA from stage 1) ==="
"$PYTHON" "$TRAIN_SCRIPT" \
  --config configs/train_stage2_miso.yaml \
  --train-file "$MISO_SFT" \
  --output-dir "$STAGE2_OUT" \
  --adapter-dir "$STAGE1_OUT"

echo ""
echo "Done. Final adapter: $STAGE2_OUT"
