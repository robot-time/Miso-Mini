#!/usr/bin/env python3
"""
Triad inference using MLX — fast on Apple Silicon (M1/M2/M3/M4).

Runs every message through all 3 stages:
  reasoning → response → critic

Usage:
  python scripts/infer_triad_mlx.py --question "What is 2+2?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlx_lm import load, generate

PROMPT_REASONING = (
    "[adapter:reasoning]\nUser question:\n{user}\n"
    "Think step by step. Show reasoning, then one short conclusion line."
)
PROMPT_RESPONSE = (
    "[adapter:response]\nUser question:\n{user}\n"
    "Reasoning draft (do not repeat long reasoning; polish into a clear helpful reply):\n{reasoning}\n"
    "Write the final user-facing reply only. Be concise and natural."
)
PROMPT_CRITIC = (
    "[adapter:critic]\nUser question:\n{user}\n"
    "Draft reply:\n{draft}\n"
    "Critique briefly, then output ONLY the improved final reply."
)

BASE_MODEL = "microsoft/Phi-4-mini-instruct"


def run_stage(adapter_dir: Path, prompt: str, max_tokens: int, *, verbose: bool) -> str:
    if verbose:
        print(f"  Loading {adapter_dir.name} adapter...", file=sys.stderr)
    model, tokenizer = load(str(BASE_MODEL), adapter_path=str(adapter_dir))
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    del model
    return response.strip()


def run_triad(adapters_dir: Path, user: str, max_tokens: int, *, verbose: bool = True) -> dict:
    user = user.strip()

    if verbose:
        print("--- Stage 1: reasoning ---", file=sys.stderr)
    reasoning = run_stage(
        adapters_dir / "reasoning", PROMPT_REASONING.format(user=user), max_tokens, verbose=verbose
    )
    if verbose:
        print(reasoning[:500], file=sys.stderr)

    if verbose:
        print("\n--- Stage 2: response ---", file=sys.stderr)
    draft = run_stage(
        adapters_dir / "response",
        PROMPT_RESPONSE.format(user=user, reasoning=reasoning),
        max_tokens,
        verbose=verbose,
    )
    if verbose:
        print(draft[:500], file=sys.stderr)

    if verbose:
        print("\n--- Stage 3: critic ---", file=sys.stderr)
    final = run_stage(
        adapters_dir / "critic",
        PROMPT_CRITIC.format(user=user, draft=draft),
        max_tokens,
        verbose=verbose,
    )

    return {"user": user, "reasoning": reasoning, "draft": draft, "final": final}


def main():
    p = argparse.ArgumentParser(description="Triad inference via MLX (Apple Silicon)")
    p.add_argument("--adapters-dir", default="outputs/triad_adapters_mlx")
    p.add_argument("--question", default="")
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--json", action="store_true", help="Print one JSON object (reasoning, draft, final) on stdout")
    args = p.parse_args()

    if not args.question.strip():
        p.error("Pass --question \"...\"")

    adapters_dir = Path(args.adapters_dir)
    result = run_triad(adapters_dir, args.question, args.max_tokens, verbose=not args.json)

    if args.json:
        print(json.dumps(result, ensure_ascii=True))
    else:
        print("\n=== Final Answer ===")
        print(result["final"])


if __name__ == "__main__":
    main()
