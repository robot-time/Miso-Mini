#!/usr/bin/env python3
"""
Scatter plot: “cost” (params, $, FLOPs — your choice) vs benchmark score.

Data: JSON file with points[] (create locally; repo does not ship example scores).

  pip install matplotlib   # or: pip install -r requirements-eval.txt

  python scripts/plot_benchmark_scatter.py \
    --input path/to/your_points.json \
    --output outputs/figures/my_scatter.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="JSON file with points[] or legacy list")
    p.add_argument("--output", default="outputs/figures/scatter.png")
    p.add_argument(
        "--title",
        default=None,
        help="Figure title (overrides JSON figure_title when set)",
    )
    args = p.parse_args()

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        points = raw
        x_label = "Cost (your metric)"
        y_label = "Score"
        highlight_name = None
        figure_title = None
        footnote = None
    else:
        points = raw["points"]
        x_label = raw.get("x_label", "Cost")
        y_label = raw.get("y_label", "Score")
        highlight_name = raw.get("highlight")
        figure_title = raw.get("figure_title")
        footnote = raw.get("footnote")

    title = args.title if args.title is not None else figure_title
    if title is None:
        title = "Performance vs cost"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for _style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        if _style in plt.style.available:
            plt.style.use(_style)
            break
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=120)

    xs, ys, names = [], [], []
    hx, hy = None, None
    for pt in points:
        name = pt["name"]
        x, y = float(pt["x"]), float(pt["y"])
        xs.append(x)
        ys.append(y)
        names.append(name)
        if highlight_name and name == highlight_name:
            hx, hy = x, y

    # Non-highlight points
    for i, (x, y, name) in enumerate(zip(xs, ys, names)):
        if highlight_name and name == highlight_name:
            continue
        ax.scatter([x], [y], s=120, c="#888888", edgecolors="white", linewidths=1, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, color="#333333")

    if hx is not None:
        ax.scatter([hx], [hy], s=220, c="#2563eb", edgecolors="white", linewidths=1.5, zorder=5, label=highlight_name)
        ax.annotate(
            highlight_name,
            (hx, hy),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            fontweight="bold",
            color="#1e40af",
        )

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, pad=12)
    if highlight_name:
        ax.legend(loc="lower right", frameon=True)

    # Leave room for optional disclaimer (mixed eval settings, etc.)
    fig.tight_layout(rect=[0, 0.14 if footnote else 0, 1, 1])
    if footnote:
        fig.text(
            0.5,
            0.02,
            footnote,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#444444",
            transform=fig.transFigure,
        )
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
