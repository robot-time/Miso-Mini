#!/usr/bin/env python3
"""
Evaluate the FULL triad (reasoning → response → critic) on a benchmark-style task.

lm-eval only loads ONE merged HF model per run, so it cannot run your 3-stage pipeline.
This script is the hook for triad-wide scores: load GSM8K (or a JSONL), run infer_triad.run_triad
per question, score with the same metrics as the task (e.g. GSM8K exact_match).

Not implemented here — wiring depends on GPU memory and whether you use greedy decoding
for comparable metrics. See docs/BENCHMARK.md ("Full triad").

Planned usage (future):

  python scripts/eval_triad_task.py \\
    --task gsm8k \\
    --adapters-dir outputs/triad_adapters \\
    --config configs/triad_reasoning.yaml \\
    --limit 100 \\
    --out eval/triad_gsm8k.jsonl
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(
        description="Placeholder: full-triad benchmark (not lm-eval). See module docstring."
    )
    p.add_argument("--task", default="gsm8k", help="Benchmark id (planned)")
    p.add_argument("--adapters-dir", default="outputs/triad_adapters")
    p.add_argument("--config", default="configs/triad_reasoning.yaml")
    p.add_argument("--limit", type=int, default=0, help="Max examples (0 = all)")
    p.add_argument("--out", default="eval/triad_runs.jsonl")
    args = p.parse_args()
    raise SystemExit(
        "eval_triad_task.py is not implemented yet. "
        "For merged single-adapter scores use: bash scripts/run_lm_eval_0shot.sh outputs/merged_reasoning. "
        "For triad behavior on a small set, use scripts/infer_triad.py --stdin or scripts/eval_triad_delta.py."
    )


if __name__ == "__main__":
    main()
