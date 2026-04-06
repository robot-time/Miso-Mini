#!/usr/bin/env python3
"""
Mix two SFT JSONL files (e.g. WildChat + Miso teacher) with a target ratio.

Each line: {"messages":[{"role":"system",...}, ...]}

Optional --tag-style prefixes system prompt for lightweight conditioning:
  casual_chat  (WildChat-heavy realism)
  reasoning    (teacher / self-refine JSON targets)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def tag_row(row: dict, tag: str) -> dict:
    out = json.loads(json.dumps(row))
    msgs = out.get("messages") or []
    if not msgs:
        return out
    prefix = f"[style:{tag}] "
    if msgs[0].get("role") == "system":
        c = msgs[0].get("content") or ""
        if not c.startswith(prefix):
            msgs[0]["content"] = prefix + c
    else:
        msgs.insert(0, {"role": "system", "content": prefix.strip()})
    return out


def sample_n(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    if n <= 0:
        return []
    if len(rows) >= n:
        return rng.sample(rows, n)
    return rng.choices(rows, k=n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wildchat", required=True, help="JSONL from wildchat_to_sft.py")
    p.add_argument("--miso", required=True, help="JSONL from convert_to_sft.py")
    p.add_argument(
        "--wildchat-ratio",
        type=float,
        default=0.7,
        help="Fraction of target rows from WildChat (rest from Miso).",
    )
    p.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Total rows after mix. Default: len(wildchat)+len(miso).",
    )
    p.add_argument("--output", default="data/mixed_sft.jsonl")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument(
        "--tag-style",
        action="store_true",
        help="Prefix system prompt: casual_chat (WC) vs reasoning (Miso).",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Shuffle final mix (default: on).",
    )
    p.add_argument("--no-shuffle", action="store_false", dest="shuffle")
    args = p.parse_args()

    if not 0.0 <= args.wildchat_ratio <= 1.0:
        raise SystemExit("wildchat-ratio must be in [0, 1]")

    wc = load_jsonl(Path(args.wildchat))
    miso = load_jsonl(Path(args.miso))
    if not wc and not miso:
        raise SystemExit("Both inputs are empty.")

    default_total = len(wc) + len(miso)
    total = args.target_size if args.target_size is not None else default_total
    total = max(1, total)

    n_wc = int(total * args.wildchat_ratio)
    n_miso = total - n_wc
    rng = random.Random(args.seed)

    pick_wc = sample_n(wc, n_wc, rng)
    pick_m = sample_n(miso, n_miso, rng)

    mixed: list[dict] = []
    for row in pick_wc:
        r = tag_row(row, "casual_chat") if args.tag_style else row
        mixed.append(r)
    for row in pick_m:
        r = tag_row(row, "reasoning") if args.tag_style else row
        mixed.append(r)

    if args.shuffle:
        rng.shuffle(mixed)

    out = Path(args.output)
    dump_jsonl(out, mixed)
    print(
        f"Wrote {len(mixed)} rows -> {out} "
        f"(wildchat picked {len(pick_wc)} / {len(wc)} avail, "
        f"miso picked {len(pick_m)} / {len(miso)} avail, target={total})"
    )


if __name__ == "__main__":
    main()
