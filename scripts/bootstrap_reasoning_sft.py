#!/usr/bin/env python3
"""Minimal reasoning-only SFT seed (CoT-style). Replace/extend with real CoT data later."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Small seed file; setup_triad_data.sh merges this with reasoning_gsm8k.jsonl when present.
OUT = ROOT / "data" / "reasoning_seed.jsonl"

SYS = (
    "[adapter:reasoning] You are the reasoning head. Think step by step. "
    "Show clear intermediate steps, then end with a brief conclusion line."
)

EXAMPLES = [
    (
        "If a train travels 60 miles in 1.5 hours, what is its average speed in mph?",
        "Step 1: Recall average speed = distance / time.\n"
        "Step 2: distance = 60 miles, time = 1.5 hours.\n"
        "Step 3: 60 / 1.5 = 40.\n"
        "Conclusion: The average speed is 40 mph.",
    ),
    (
        "Is 441 divisible by 7? Show reasoning.",
        "Step 1: Divide 441 by 7.\n"
        "Step 2: 7 × 63 = 441.\n"
        "Conclusion: Yes, 441 is divisible by 7.",
    ),
    (
        "Why might cross-validation give a more reliable estimate than a single train/test split?",
        "Step 1: A single split depends on which points landed in train vs test.\n"
        "Step 2: Cross-validation averages performance over multiple splits.\n"
        "Step 3: That reduces variance from one lucky/unlucky partition.\n"
        "Conclusion: Cross-validation is usually more reliable for estimating generalization.",
    ),
    (
        "What is the main risk of overfitting?",
        "Step 1: Overfitting means the model fits training noise, not just signal.\n"
        "Step 2: That hurts performance on new data.\n"
        "Conclusion: The main risk is poor generalization to unseen examples.",
    ),
    (
        "Compare AND vs OR in boolean logic in one short example.",
        "Step 1: AND is true only if both inputs are true.\n"
        "Step 2: OR is true if at least one input is true.\n"
        "Example: True AND False = False; True OR False = True.\n"
        "Conclusion: AND is stricter than OR.",
    ),
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for q, a in EXAMPLES:
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            }
        )
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    print(f"Wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
