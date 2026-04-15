#!/usr/bin/env python3
"""
Bar charts from an lm-evaluation-harness output tree (e.g. benchmark_suite_* or RunPod v2_* folders).

Finds each immediate subfolder (gsm8k_0shot, arc_challenge_10shot, …), then the nested
``results_<timestamp>.json``. Uses ``manifest.json`` run order when present.

  pip install -r requirements-eval.txt   # matplotlib

  python scripts/plot_lm_eval_suite.py \\
    eval/runpod_v2_benchmark_20260406_054840/v2_benchmark_three_20260406_054840 \\
    --output-dir eval/runpod_v2_benchmark_20260406_054840/figures

  python scripts/plot_lm_eval_suite.py eval/benchmark_suite_2026-01-01/ \\
    --title \"Merged reasoning LoRA\"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _use_style() -> None:
    import matplotlib.pyplot as plt

    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        if name in plt.style.available:
            plt.style.use(name)
            return


def _find_results_json(run_dir: Path) -> Path | None:
    matches = sorted(run_dir.rglob("results_*.json"))
    return matches[0] if matches else None


def _stderr_for(metric_key: str, task_blob: dict) -> float | None:
    base, _, tail = metric_key.partition(",")
    if not tail:
        return None
    cand = f"{base}_stderr,{tail}"
    v = task_blob.get(cand)
    return float(v) if v is not None else None


def pick_headline_metric(task_name: str, task_blob: dict) -> tuple[str, float, float | None]:
    """
    Return (metric_label, value 0..1, stderr or None).
    """
    keys = set(task_blob.keys())
    # GSM8K-style generative exact match
    if "exact_match,flexible-extract" in keys:
        k = "exact_match,flexible-extract"
        return ("exact match (flex)", float(task_blob[k]), _stderr_for(k, task_blob))
    if "exact_match,strict-match" in keys:
        k = "exact_match,strict-match"
        return ("exact match (strict)", float(task_blob[k]), _stderr_for(k, task_blob))
    # Multiple-choice style
    if "acc_norm,none" in keys:
        k = "acc_norm,none"
        return ("acc_norm", float(task_blob[k]), _stderr_for(k, task_blob))
    if "acc,none" in keys:
        k = "acc,none"
        return ("acc", float(task_blob[k]), _stderr_for(k, task_blob))
    # MMLU / BBH aggregates sometimes expose acc,none on group key
    for k, v in task_blob.items():
        if k.endswith(",none") and not k.endswith("_stderr,none") and isinstance(v, (int, float)):
            if 0 <= float(v) <= 1 and "stderr" not in k:
                return (k.split(",")[0], float(v), _stderr_for(k, task_blob))
    raise ValueError(f"No known headline metric in results for {task_name}: {sorted(keys)[:12]}…")


def _manifest_runs(suite_dir: Path) -> list[dict] | None:
    man = suite_dir / "manifest.json"
    if not man.is_file():
        return None
    data = json.loads(man.read_text(encoding="utf-8"))
    runs = data.get("runs")
    if not isinstance(runs, list):
        return None
    return [r for r in runs if isinstance(r, dict) and r.get("id")]


def _run_order(suite_dir: Path) -> list[str] | None:
    runs = _manifest_runs(suite_dir)
    if not runs:
        return None
    return [str(r["id"]) for r in runs]


def _pretty_run_label(run_id: str) -> str:
    s = run_id.replace("_", " ")
    s = re.sub(r"(\d+)shot", r"\1-shot", s)
    return s.title()


def _pick_task_name(res: dict, *, preferred: str | None) -> str:
    if len(res) == 1:
        return next(iter(res))
    if preferred and preferred in res:
        return preferred
    # e.g. MMLU group: prefer aggregate-like keys (short alias) over per-subject
    keys = list(res.keys())
    for k in keys:
        if "," not in k and k.count("_") <= 2:
            return k
    return sorted(keys)[0]


def collect_rows(suite_dir: Path) -> list[dict]:
    order = _run_order(suite_dir)
    manifest = _manifest_runs(suite_dir)
    task_hint: dict[str, str] = {}
    if manifest:
        for r in manifest:
            rid = str(r["id"])
            t = r.get("tasks")
            if isinstance(t, str) and "," not in t:
                task_hint[rid] = t.strip()

    children = [p for p in suite_dir.iterdir() if p.is_dir() and p.name != "__pycache__"]
    by_name = {p.name: p for p in children}

    if order:
        dirs = [by_name[n] for n in order if n in by_name]
        dirs += [p for p in children if p.name not in set(order)]
    else:
        dirs = sorted(children, key=lambda p: p.name)

    rows: list[dict] = []
    for d in dirs:
        rj = _find_results_json(d)
        if rj is None:
            continue
        data = json.loads(rj.read_text(encoding="utf-8"))
        res = data.get("results") or {}
        pref = task_hint.get(d.name)
        task_name = _pick_task_name(res, preferred=pref)
        blob = res[task_name]
        mlabel, value, stderr = pick_headline_metric(task_name, blob)
        nshot = (data.get("n-shot") or {}).get(task_name)
        rows.append(
            {
                "run_id": d.name,
                "label": _pretty_run_label(d.name),
                "task": task_name,
                "metric": mlabel,
                "value": value,
                "stderr": stderr,
                "results_path": str(rj),
            }
        )
    return rows


def write_summary_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def plot_bars(rows: list[dict], out_png: Path, title: str | None) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        raise SystemExit("No lm-eval result folders found under suite directory.")

    _use_style()
    labels = [r["label"] for r in rows]
    ys = [100.0 * r["value"] for r in rows]
    yerr = []
    for r in rows:
        s = r["stderr"]
        yerr.append(100.0 * s if s is not None else 0.0)
    has_err = any(e > 0 for e in yerr)

    fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * len(rows)), 5), dpi=120)
    x = range(len(rows))
    colors = ["#2563eb"] * len(rows)
    bars = ax.bar(
        list(x),
        ys,
        yerr=yerr if has_err else None,
        capsize=4 if has_err else 0,
        color=colors,
        edgecolor="white",
        linewidth=1,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, min(105, max(ys + [10]) * 1.15))
    ttl = title or "lm-eval headline metrics"
    ax.set_title(ttl, fontsize=12, pad=12)

    # Value labels on bars
    for rect, y, r in zip(bars, ys, rows):
        sub = f'{r["metric"]}'
        ax.annotate(
            f"{y:.1f}%\n({sub})",
            xy=(rect.get_x() + rect.get_width() / 2, y),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1e293b",
        )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot bar chart from lm-eval suite output directory.")
    p.add_argument(
        "suite_dir",
        type=Path,
        help="Directory containing per-run subfolders (each with nested results_*.json)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write figures/ (default: <suite_dir>/figures)",
    )
    p.add_argument("--title", default=None, help="Figure title")
    args = p.parse_args()

    suite_dir = args.suite_dir.resolve()
    if not suite_dir.is_dir():
        raise SystemExit(f"Not a directory: {suite_dir}")

    out_dir = (args.output_dir or (suite_dir / "figures")).resolve()
    rows = collect_rows(suite_dir)
    write_summary_json(rows, out_dir / "suite_plot_summary.json")
    stem = suite_dir.name.replace(" ", "_")
    plot_bars(rows, out_dir / f"{stem}_scores.png", args.title)


if __name__ == "__main__":
    main()
