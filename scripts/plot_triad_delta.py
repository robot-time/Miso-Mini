#!/usr/bin/env python3
"""
Plot triad “critic effect” from eval_triad_delta.py JSONL output.

Shows: average draft vs final length, % of rows where draft ≠ final.

  python scripts/eval_triad_delta.py ... --out eval/triad_runs.jsonl
  python scripts/plot_triad_delta.py --input eval/triad_runs.jsonl --output outputs/figures/triad_critic_delta.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="JSONL from eval_triad_delta.py")
    p.add_argument("--output", default="outputs/figures/triad_critic_delta.png")
    args = p.parse_args()

    rows = load_jsonl(Path(args.input))
    if not rows:
        raise SystemExit("No rows in input")

    changed = sum(1 for r in rows if r.get("draft_ne_final"))
    n = len(rows)
    avg_draft = sum(len(r.get("draft", "")) for r in rows) / n
    avg_final = sum(len(r.get("final", "")) for r in rows) / n

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for _style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        if _style in plt.style.available:
            plt.style.use(_style)
            break
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=120)

    # Left: draft vs final length
    ax0 = axes[0]
    ax0.bar(["Draft (response)", "Final (critic)"], [avg_draft, avg_final], color=["#94a3b8", "#2563eb"], edgecolor="white")
    ax0.set_ylabel("Mean character length")
    ax0.set_title("Length after each stage")

    # Right: % changed by critic
    ax1 = axes[1]
    pct_same = 100 * (n - changed) / n
    pct_changed = 100 * changed / n
    ax1.bar(
        ["Unchanged", "Critic edited"],
        [pct_same, pct_changed],
        color=["#cbd5e1", "#7c3aed"],
        edgecolor="white",
    )
    ax1.set_ylabel("Share of items (%)")
    ax1.set_title("Did critic change the draft?")
    ax1.set_ylim(0, 105)

    fig.suptitle("Triad critic effect (your differentiator)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out} (n={n}, avg_draft={avg_draft:.0f}, avg_final={avg_final:.0f}, changed={changed}/{n})")


if __name__ == "__main__":
    main()
