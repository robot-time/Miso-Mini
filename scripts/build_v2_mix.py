#!/usr/bin/env python3
"""
Build V2 balanced SFT JSONL (~40% general chat, ~30% math, ~30% hard reasoning).

- General: WildChat-1M first-turn (reuse scripts/wildchat_to_sft.py helpers).
- Math: GSM8K train parquet — half chain-of-thought style, half direct-final-answer only.
- Hard: ARC-Challenge + ARC-Easy **train** + BBH (lukaemon/bbh **test** shards; may overlap lm-eval BBH — use for format practice, not clean leaderboard).

Output lines match train_unsloth.py: {"messages":[{role,content},...]}

  python scripts/build_v2_mix.py --output data/v2_mix_sft.jsonl --total-rows 12000

Requires: pandas, huggingface_hub, tqdm (and wildchat_to_sft.py on path via importlib).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
from pathlib import Path

# --- prompts (two-format mix) ---
SYS_CHAT = (
    "You are a helpful assistant. Follow the user's instructions. "
    "Be accurate and concise unless the user asks for detail."
)

SYS_COT = (
    "Think step by step. After your reasoning, give the final answer clearly "
    "on the last line."
)

SYS_DIRECT = (
    "Give ONLY the final answer. No explanation, no preamble. "
    "For multiple choice respond with a single letter (A–D) when applicable. "
    "For numeric tasks end with a line: #### <number>"
)

GSM8K_TRAIN_PARQUET = (
    "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet"
)

ARC_REVISION = "refs/convert/parquet"
ARC_REPO = "allenai/ai2_arc"
BBH_REPO = "lukaemon/bbh"

BBH_TASKS = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
)


def _load_wildchat_module():
    path = Path(__file__).resolve().parent / "wildchat_to_sft.py"
    spec = importlib.util.spec_from_file_location("wildchat_to_sft", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def gsm8k_final_line(answer: str) -> str:
    s = answer.strip()
    m = re.search(r"####\s*([^\n]+)", s)
    if m:
        return f"#### {m.group(1).strip()}"
    last = s.split("\n")[-1].strip()
    return last if last else s


def _to_plain_list(val):
    """Parquet/pyarrow may give numpy arrays; avoid ambiguous truthiness."""
    if val is None:
        return []
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    return [str(val)]


def arc_format_choices(choices) -> str:
    if choices is None:
        return ""
    if hasattr(choices, "tolist") and not isinstance(choices, dict):
        choices = choices.tolist()
    if isinstance(choices, dict):
        texts = _to_plain_list(choices.get("text"))
        labels_raw = choices.get("label")
        labels = _to_plain_list(labels_raw) if labels_raw is not None else []
        if not labels:
            labels = [chr(65 + i) for i in range(len(texts))]
        lines = []
        for i, t in enumerate(texts):
            lab = labels[i] if i < len(labels) else chr(65 + i)
            lines.append(f"{lab}) {t}")
        return "\n".join(lines)
    return str(choices)


def message_row(system: str, user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def collect_general(n: int, rng: random.Random, wildchat_repo: str) -> list[dict]:
    if n <= 0:
        return []
    wc = _load_wildchat_module()
    out: list[dict] = []
    scanned = 0
    max_scan = max(n * 50, n + 5000)
    for row in wc.iter_parquet_shards(wildchat_repo, max_conversations=max_scan):
        scanned += 1
        ex = wc.build_first_turn_example(
            wc.row_to_conversation(row) or [], SYS_CHAT
        )
        if ex:
            out.append(ex)
        if len(out) >= n:
            break
    if len(out) < n:
        print(
            f"WARNING: general bucket wanted {n}, got {len(out)} (WildChat scan {scanned})",
            flush=True,
        )
    rng.shuffle(out)
    return out[:n]


def collect_gsm8k(n: int, rng: random.Random, parquet_url: str) -> list[dict]:
    if n <= 0:
        return []
    import pandas as pd

    df = pd.read_parquet(parquet_url)
    if "question" not in df.columns or "answer" not in df.columns:
        raise SystemExit(f"GSM8K parquet missing columns: {list(df.columns)}")
    idx = list(range(len(df)))
    rng.shuffle(idx)
    out: list[dict] = []
    for i in range(n):
        row_i = idx[i % len(idx)]
        q = str(df.iloc[row_i]["question"]).strip()
        a_full = str(df.iloc[row_i]["answer"]).strip()
        if not q or not a_full:
            continue
        use_cot = i % 2 == 0
        if use_cot:
            user = f"Solve this step by step.\n\n{q}"
            out.append(message_row(SYS_COT, user, a_full))
        else:
            user = (
                "Solve the problem. Give ONLY the final answer as one line: #### <integer>\n\n"
                f"{q}"
            )
            out.append(message_row(SYS_DIRECT, user, gsm8k_final_line(a_full)))
    rng.shuffle(out)
    return out


def collect_arc(
    n: int,
    rng: random.Random,
    hf_download,
) -> list[dict]:
    if n <= 0:
        return []
    import pandas as pd

    pc = hf_download(ARC_REPO, "ARC-Challenge/train/0000.parquet", ARC_REVISION)
    pe = hf_download(ARC_REPO, "ARC-Easy/train/0000.parquet", ARC_REVISION)
    dfc = pd.read_parquet(pc)
    dfe = pd.read_parquet(pe)

    def pack(df, i: int, use_cot: bool) -> dict | None:
        r = df.iloc[i]
        q = str(r.get("question", "")).strip()
        key = str(r.get("answerKey", "")).strip()
        ch = arc_format_choices(r.get("choices"))
        if not q or not key:
            return None
        body = f"Question:\n{q}\n\nChoices:\n{ch}"
        if use_cot:
            user = f"{body}\n\nThink step by step, then end with exactly: Answer: {key}"
            assistant = (
                f"Working through the options, the best supported answer is ({key}).\n"
                f"Answer: {key}"
            )
            return message_row(SYS_COT, user, assistant)
        user = f"Give ONLY the answer letter (A, B, C, or D). No other text.\n\n{body}"
        return message_row(SYS_DIRECT, user, key)

    candidates: list[dict] = []
    for df in (dfc, dfe):
        for i in range(len(df)):
            for use_cot in (True, False):
                m = pack(df, i, use_cot)
                if m:
                    candidates.append(m)
    if not candidates:
        return []
    rng.shuffle(candidates)
    if len(candidates) >= n:
        return candidates[:n]
    # With replacement if pool smaller than n
    return [rng.choice(candidates) for _ in range(n)]


def collect_bbh(n: int, rng: random.Random, hf_download) -> list[dict]:
    if n <= 0:
        return []
    import pandas as pd

    per_task = max(1, n // len(BBH_TASKS))
    out: list[dict] = []
    for task in BBH_TASKS:
        if len(out) >= n:
            break
        path = f"{task}/test-00000-of-00001.parquet"
        try:
            p = hf_download(BBH_REPO, path, None)
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"WARNING: skip BBH {task}: {e}", flush=True)
            continue
        take = min(per_task, len(df), n - len(out))
        idxs = list(range(len(df)))
        rng.shuffle(idxs)
        for i in idxs[:take]:
            inp = str(df.iloc[i]["input"]).strip()
            tgt = str(df.iloc[i]["target"]).strip()
            if not inp or not tgt:
                continue
            use_cot = rng.random() < 0.5
            if use_cot:
                user = (
                    "Solve the task. Think briefly, then give ONLY the final answer "
                    "exactly in the format expected by the task (same as the reference).\n\n"
                    + inp[:12000]
                )
                assistant = f"Reasoning:\nBrief work toward the solution.\n\nFinal:\n{tgt}"
                out.append(message_row(SYS_COT, user, assistant))
            else:
                user = (
                    "Give ONLY the final answer. No explanation.\n\n" + inp[:12000]
                )
                out.append(message_row(SYS_DIRECT, user, tgt))
    rng.shuffle(out)
    return out[:n]


def main() -> None:
    p = argparse.ArgumentParser(description="Build V2 balanced SFT mix JSONL.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--total-rows", type=int, default=12000)
    p.add_argument("--general-frac", type=float, default=0.4)
    p.add_argument("--math-frac", type=float, default=0.3)
    p.add_argument("--hard-frac", type=float, default=0.3)
    p.add_argument("--wildchat-repo", default="allenai/WildChat-1M")
    p.add_argument("--gsm8k-parquet", default=GSM8K_TRAIN_PARQUET)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    fr = args.general_frac + args.math_frac + args.hard_frac
    if abs(fr - 1.0) > 1e-6:
        raise SystemExit(f"Fractions must sum to 1.0, got {fr}")

    rng = random.Random(args.seed)
    n = args.total_rows
    n_gen = int(round(n * args.general_frac))
    n_math = int(round(n * args.math_frac))
    n_hard = n - n_gen - n_math

    from huggingface_hub import hf_hub_download

    def _dl(repo: str, fn: str, rev: str | None):
        kw = dict(repo_id=repo, filename=fn, repo_type="dataset")
        if rev:
            kw["revision"] = rev
        return hf_hub_download(**kw)

    print(
        f"Targets: general={n_gen} math={n_math} hard={n_hard} (total {n_gen + n_math + n_hard})",
        flush=True,
    )

    hard_arc = n_hard // 2
    hard_bbh = n_hard - hard_arc

    rows_g = collect_general(n_gen, rng, args.wildchat_repo)
    rows_m = collect_gsm8k(n_math, rng, args.gsm8k_parquet)
    rows_a = collect_arc(hard_arc, rng, _dl)
    rows_b = collect_bbh(hard_bbh, rng, _dl)

    # If BBH or ARC under-filled, top up from ARC/GSM8K
    merged = rows_g + rows_m + rows_a + rows_b
    if len(merged) < n:
        need = n - len(merged)
        print(f"Padding {need} rows (GSM8K)", flush=True)
        merged.extend(collect_gsm8k(need, rng, args.gsm8k_parquet))
    if len(merged) < n:
        need = n - len(merged)
        print(f"Padding {need} rows (WildChat)", flush=True)
        merged.extend(collect_general(need, rng, args.wildchat_repo))
    merged = merged[:n]
    rng.shuffle(merged)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote {len(merged)} rows -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
