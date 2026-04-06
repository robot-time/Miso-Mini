#!/usr/bin/env python3
"""
Run the triad (MLX) on a JSONL of questions and report draft vs final stats.

Each line: {"id": "1", "question": "..."}  (id optional)

Metrics: changed (draft != final), length ratio, optional gold substring match.

  python scripts/eval_triad_delta.py --questions data/eval_questions_sample.jsonl --out eval/triad_runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from infer_triad_mlx import run_triad


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
    p.add_argument("--questions", required=True, help="JSONL with question field")
    p.add_argument("--adapters-dir", default="outputs/triad_adapters_mlx")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--out", default="", help="Write one JSON result per line")
    args = p.parse_args()

    adapters_dir = Path(args.adapters_dir)
    rows = load_jsonl(Path(args.questions))
    out_f = open(args.out, "w", encoding="utf-8") if args.out else None

    agg = {"n": 0, "changed": 0, "draft_len": 0, "final_len": 0, "gold_hit": 0}
    gold_n = 0

    for row in rows:
        q = row.get("question", "").strip()
        if not q:
            continue
        rid = row.get("id", "")
        gold = (row.get("gold") or "").strip()

        r = run_triad(adapters_dir, q, args.max_tokens, verbose=False)
        draft, final = r["draft"].strip(), r["final"].strip()
        changed = draft != final
        agg["n"] += 1
        agg["changed"] += int(changed)
        agg["draft_len"] += len(draft)
        agg["final_len"] += len(final)
        if gold:
            gold_n += 1
            if gold.lower() in final.lower():
                agg["gold_hit"] += 1

        rec = {
            "id": rid,
            "question": q,
            "reasoning": r["reasoning"],
            "draft": r["draft"],
            "final": r["final"],
            "draft_ne_final": changed,
        }
        if gold:
            rec["gold_in_final"] = gold.lower() in final.lower()

        line = json.dumps(rec, ensure_ascii=True)
        print(line)
        if out_f:
            out_f.write(line + "\n")

    if out_f:
        out_f.close()

    if agg["n"]:
        print(
            json.dumps(
                {
                    "samples": agg["n"],
                    "pct_draft_changed_by_critic": round(100 * agg["changed"] / agg["n"], 1),
                    "avg_draft_chars": round(agg["draft_len"] / agg["n"], 1),
                    "avg_final_chars": round(agg["final_len"] / agg["n"], 1),
                    "gold_hit_rate": round(100 * agg["gold_hit"] / gold_n, 1) if gold_n else None,
                },
                indent=2,
            ),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
