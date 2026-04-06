#!/usr/bin/env python3
"""
Export GSM8K (or compatible parquet) to triad reasoning SFT JSONL — same schema as bootstrap_reasoning_sft.py.

Default: downloads train split from Hugging Face parquet (no `datasets.load_dataset`, so it works on
Python 3.14 where `datasets` can hit pickler bugs).

Usage:
  python scripts/gsm8k_to_reasoning_sft.py --output data/reasoning_gsm8k.jsonl
  python scripts/gsm8k_to_reasoning_sft.py --max-rows 2000 --append --output data/reasoning_sft.jsonl

Merge with seed:
  python scripts/merge_jsonl.py --output data/reasoning_sft.jsonl data/reasoning_seed.jsonl data/reasoning_gsm8k.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Match scripts/bootstrap_reasoning_sft.py
SYS = (
    "[adapter:reasoning] You are the reasoning head. Think step by step. "
    "Show clear intermediate steps, then end with a brief conclusion line."
)

DEFAULT_TRAIN_PARQUET = (
    "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet"
)


def row_to_messages(question: str, answer: str) -> dict:
    user = f"Solve this step by step.\n\n{question.strip()}"
    return {
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer.strip()},
        ]
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--parquet",
        type=str,
        default=DEFAULT_TRAIN_PARQUET,
        help="Local path or https URL to a parquet with question + answer columns (default: GSM8K train)",
    )
    p.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    p.add_argument("--max-rows", type=int, default=0, help="Cap rows (0 = all)")
    p.add_argument(
        "--append",
        action="store_true",
        help="Append to output file instead of overwriting",
    )
    args = p.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.parquet)
    if "question" not in df.columns or "answer" not in df.columns:
        raise SystemExit(
            f"Expected columns question, answer; got {list(df.columns)}"
        )

    n = len(df)
    if args.max_rows and args.max_rows > 0:
        n = min(n, args.max_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    written = 0
    with args.output.open(mode, encoding="utf-8") as f:
        for i in range(n):
            q = df.iloc[i]["question"]
            a = df.iloc[i]["answer"]
            if pd.isna(q) or pd.isna(a):
                continue
            rec = row_to_messages(str(q), str(a))
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            written += 1

    print(f"Wrote {written} rows -> {args.output}")


if __name__ == "__main__":
    main()
