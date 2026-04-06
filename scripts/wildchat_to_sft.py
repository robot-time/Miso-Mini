#!/usr/bin/env python3
"""
Build SFT JSONL from WildChat-style HF datasets — no OpenAI calls.

The penfever/wildchat-50m repo (https://github.com/penfever/wildchat-50m) mainly:
  - loads a dataset with a `conversation` column
  - runs vLLM on those prompts to *replace* assistant text with a local model

That saves API $ but still costs GPU time. This script uses the *existing* assistant
messages already in the dataset (e.g. allenai/WildChat-1M) as supervision — free
aside from disk/network. License: follow the dataset card (WildChat is ODC-BY).

Output format matches scripts/convert_to_sft.py (chat messages) so train_unsloth works.

Python 3.14 note: `datasets.load_dataset` can hit a pickle/dill bug in the HF
builder. We fall back to downloading parquet shards from `refs/convert/parquet`
via huggingface_hub + pandas (no DatasetBuilder).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tqdm import tqdm

SYSTEM_BASE = (
    "You are a concise reasoning assistant. Think in structured steps and "
    "provide a refined final response."
)


def _norm_content(val) -> str:
    if val is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(val):
            return ""
    except Exception:
        pass
    if isinstance(val, float) and (math.isnan(val) or val != val):
        return ""
    return str(val).strip()


def turns_to_messages(conv: list) -> list[dict]:
    """Normalize conversation items to {role, content} (case-insensitive roles; parquet-safe)."""
    out = []
    for t in conv:
        if not isinstance(t, dict):
            continue
        role_raw = t.get("role")
        role = str(role_raw).strip().lower() if role_raw is not None else ""
        if role in ("human", "user"):
            role = "user"
        elif role == "assistant":
            role = "assistant"
        else:
            continue
        content = _norm_content(t.get("content"))
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def build_first_turn_example(conv: list, system_text: str) -> dict | None:
    msgs = turns_to_messages(conv)
    for i in range(len(msgs) - 1):
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            return {
                "messages": [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": msgs[i]["content"]},
                    {"role": "assistant", "content": msgs[i + 1]["content"]},
                ]
            }
    return None


def build_all_turn_examples(conv: list, system_text: str) -> list[dict]:
    """One SFT row per (prefix through user_t) -> assistant_t, WildChat-style."""
    msgs = turns_to_messages(conv)
    rows = []
    for i in range(len(msgs) - 1):
        if msgs[i]["role"] != "user" or msgs[i + 1]["role"] != "assistant":
            continue
        prefix = msgs[: i + 1]
        reply = msgs[i + 1]["content"]
        chat = [{"role": "system", "content": system_text}]
        for m in prefix:
            chat.append({"role": m["role"], "content": m["content"]})
        chat.append({"role": "assistant", "content": reply})
        rows.append({"messages": chat})
    return rows


def row_to_conversation(row) -> list | None:
    conv = row.get("conversation") if isinstance(row, dict) else None
    if conv is None and hasattr(row, "get"):
        conv = row.get("conversation")
    if hasattr(conv, "tolist"):
        conv = conv.tolist()
    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except json.JSONDecodeError:
            return None
    if not isinstance(conv, list):
        return None
    return conv


def _process_row(
    row: dict,
    mode: str,
    skip_toxic: bool,
    out_f,
    system_text: str,
) -> tuple[int, int]:
    """Returns (lines_written, lines_scanned increment)."""
    if skip_toxic and bool(row.get("toxic")):
        return 0, 1
    conv = row_to_conversation(row)
    if not conv:
        return 0, 1
    n = 0
    if mode == "first_turn":
        ex = build_first_turn_example(conv, system_text)
        if ex:
            out_f.write(json.dumps(ex, ensure_ascii=True) + "\n")
            n = 1
    else:
        for ex in build_all_turn_examples(conv, system_text):
            out_f.write(json.dumps(ex, ensure_ascii=True) + "\n")
            n += 1
    return n, 1


def iter_parquet_shards(repo_id: str, max_conversations: int):
    """Yield conversation rows from Hub parquet shards (refs/convert/parquet)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    revision = "refs/convert/parquet"
    scanned = 0
    shard = 0
    while scanned < max_conversations:
        fn = f"default/train/{shard:04d}.parquet"
        try:
            path = hf_hub_download(
                repo_id,
                filename=fn,
                repo_type="dataset",
                revision=revision,
            )
        except Exception as e:
            if shard == 0:
                raise RuntimeError(
                    f"Failed to download parquet {fn} from {repo_id}@{revision}: {e}"
                ) from e
            break
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            yield row.to_dict()
            scanned += 1
            if scanned >= max_conversations:
                return
        shard += 1
        if shard > 200:
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        default="allenai/WildChat-1M",
        help="HF dataset id with `conversation` column (parquet fallback: allenai/WildChat-1M).",
    )
    p.add_argument("--split", default="train")
    p.add_argument(
        "--mode",
        choices=("first_turn", "all_turns"),
        default="first_turn",
        help="first_turn: one example per conv; all_turns: one per user/assistant pair.",
    )
    p.add_argument("--max-rows", type=int, default=5000, help="Max conversations to scan.")
    p.add_argument(
        "--streaming",
        action="store_true",
        help="Stream dataset (low RAM). Ignored if --parquet-only.",
    )
    p.add_argument(
        "--parquet-only",
        action="store_true",
        help="Skip datasets.load_dataset; use Hub parquet shards only (recommended on Python 3.14+).",
    )
    p.add_argument(
        "--skip-toxic",
        action="store_true",
        dest="skip_toxic",
        help="Skip rows where column `toxic` is True (if present).",
    )
    p.add_argument(
        "--output",
        default="data/wildchat_sft.jsonl",
    )
    p.add_argument(
        "--system-prefix",
        default="",
        help="Prepended to system prompt (e.g. [adapter:response] for triad response head).",
    )
    args = p.parse_args()

    system_text = SYSTEM_BASE
    if args.system_prefix.strip():
        system_text = args.system_prefix.strip() + " " + SYSTEM_BASE

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = max(1, args.max_rows)
    written = 0
    scanned = 0

    def run_with_iterator(it, total: int | None):
        nonlocal written, scanned
        with out_path.open("w", encoding="utf-8") as f:
            for row in tqdm(it, total=total, desc="WildChat→SFT"):
                w, s = _process_row(row, args.mode, args.skip_toxic, f, system_text)
                written += w
                scanned += s

    if args.parquet_only:
        if args.dataset != "allenai/WildChat-1M":
            raise SystemExit(
                "--parquet-only supports allenai/WildChat-1M (Hub parquet layout under refs/convert/parquet)."
            )
        run_with_iterator(iter_parquet_shards(args.dataset, cap), cap)
        print(f"Scanned ~{scanned} conversations, wrote {written} SFT rows -> {out_path}")
        return

    use_parquet = False
    it = None
    bar_total = None
    try:
        from datasets import load_dataset

        if args.streaming:
            ds = load_dataset(args.dataset, split=args.split, streaming=True)
            it = ds.take(cap)
            bar_total = cap
        else:
            sliced = f"{args.split}[:{cap}]"
            ds = load_dataset(args.dataset, split=sliced)
            it = ds
            bar_total = len(ds)
    except Exception as e:
        print(
            f"[warn] datasets.load_dataset failed ({type(e).__name__}: {e}). "
            "Using parquet Hub fallback (same data, no DatasetBuilder)."
        )
        use_parquet = True

    if use_parquet:
        run_with_iterator(iter_parquet_shards(args.dataset, cap), cap)
    else:
        run_with_iterator(it, bar_total)

    print(f"Scanned ~{scanned} conversations, wrote {written} SFT rows -> {out_path}")


if __name__ == "__main__":
    main()
