#!/usr/bin/env python3
"""
Merge base model + one PEFT LoRA into a full HF folder for lm-eval / standard benchmarks.

Use the **reasoning** adapter for GSM8K-style tasks; response/critic are chat-oriented.

  python scripts/merge_adapter_for_eval.py \
    --adapter outputs/triad_adapters/reasoning \
    --output outputs/merged_reasoning

Then (with GPU, after: pip install -r requirements-eval.txt):

  lm_eval --model hf --model_args pretrained=outputs/merged_reasoning \
    --tasks gsm8k --device cuda --batch_size auto
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="microsoft/Phi-4-mini-instruct", help="Base model id or path")
    p.add_argument("--adapter", required=True, help="PEFT adapter dir (e.g. outputs/triad_adapters/reasoning)")
    p.add_argument("--output", required=True, help="Merged model output directory")
    p.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Weight dtype for merge (float16 is usual for eval on GPU)",
    )
    args = p.parse_args()

    adapter = Path(args.adapter)
    out = Path(args.output)
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)

    dt = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base on {device} ({args.dtype})...")

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=dt,
        low_cpu_mem_usage=True,
    )
    if device == "cuda":
        model = model.to("cuda")

    print(f"Merging adapter {adapter}...")
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model = model.merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(str(out))
    print(f"Merged model saved -> {out}")


if __name__ == "__main__":
    main()
