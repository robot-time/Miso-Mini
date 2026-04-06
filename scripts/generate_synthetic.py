import argparse
import json
import os
import random
import uuid
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

MODEL_PRICING_USD_PER_1M = {
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
}


def read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def make_teacher_messages(task_type: str, seed: dict, prompts_dir: Path):
    prompt_map = {
        "qa_refine": "teacher_refine.txt",
        "debate": "teacher_debate.txt",
        "compression": "teacher_compression.txt",
    }
    system_prompt = read_prompt(prompts_dir / prompt_map[task_type])
    user_prompt = json.dumps(
        {
            "question": seed.get("question", ""),
            "context": seed.get("context", ""),
            "hint_domain": seed.get("domain", "general"),
        },
        ensure_ascii=True,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _first_balanced_json_object(text: str) -> str | None:
    """Extract first top-level {...} slice; handles strings with braces."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def parse_teacher_json(raw: str | None) -> dict:
    if raw is None or not str(raw).strip():
        raise ValueError("empty model content")
    text = _strip_markdown_fences(str(raw).strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        blob = _first_balanced_json_object(text)
        if blob is None:
            raise ValueError("no JSON object found") from None
        return json.loads(blob)


def usage_cost_usd(usage, in_per_1m: float, out_per_1m: float) -> float:
    if not usage:
        return 0.0
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    in_cost = (prompt_tokens / 1_000_000.0) * in_per_1m
    out_cost = (completion_tokens / 1_000_000.0) * out_per_1m
    return in_cost + out_cost


def load_api_key_from_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return line
    return None


def resolve_openai_api_key(key_file: Path) -> str:
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    resolved = key_file if key_file.is_absolute() else Path.cwd() / key_file
    from_file = load_api_key_from_file(resolved)
    if from_file:
        return from_file.strip()
    raise RuntimeError(
        "No API key: set OPENAI_API_KEY or put your key on one line in "
        f"{resolved} (see openai_key.example)."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--teacher-model", default="gpt-5.4-mini")
    parser.add_argument("--samples-per-seed", type=int, default=1)
    parser.add_argument(
        "--mix",
        default="qa_refine:0.6,debate:0.2,compression:0.2",
        help="comma-separated task_type:weight",
    )
    parser.add_argument("--prompts-dir", default="prompts")
    parser.add_argument(
        "--key-file",
        default="openai_key.txt",
        help="Plain-text file with API key on first non-comment line (if env unset).",
    )
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument("--budget-usd", type=float, default=30.0)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new rows to out-file instead of overwriting.",
    )
    parser.add_argument(
        "--shuffle-seeds",
        action="store_true",
        help="Shuffle seed order each run (more variety across budgeted batches).",
    )
    parser.add_argument(
        "--input-price-per-1m",
        type=float,
        default=None,
        help="Override model input token price in USD per 1M tokens.",
    )
    parser.add_argument(
        "--output-price-per-1m",
        type=float,
        default=None,
        help="Override model output token price in USD per 1M tokens.",
    )
    args = parser.parse_args()

    api_key = resolve_openai_api_key(Path(args.key_file))

    seed_rows = load_jsonl(Path(args.seed_file))
    if args.shuffle_seeds:
        random.shuffle(seed_rows)
    prompts_dir = Path(args.prompts_dir)

    mix_parts = [x.strip() for x in args.mix.split(",") if x.strip()]
    task_types = []
    weights = []
    for part in mix_parts:
        t, w = part.split(":")
        task_types.append(t)
        weights.append(float(w))

    client = OpenAI(api_key=api_key)
    new_rows = []
    out_path = Path(args.out_file)
    prior_n = 0
    if args.append and out_path.is_file():
        with out_path.open("r", encoding="utf-8") as f:
            prior_n = sum(1 for line in f if line.strip())
    pricing = MODEL_PRICING_USD_PER_1M.get(args.teacher_model, {})
    input_price = (
        args.input_price_per_1m
        if args.input_price_per_1m is not None
        else pricing.get("input", 0.0)
    )
    output_price = (
        args.output_price_per_1m
        if args.output_price_per_1m is not None
        else pricing.get("output", 0.0)
    )
    if input_price <= 0.0 or output_price <= 0.0:
        raise RuntimeError(
            "Missing pricing for this model. Pass --input-price-per-1m and "
            "--output-price-per-1m explicitly."
        )
    running_cost = 0.0
    budget_hit = False
    warned_missing_usage = False
    skipped_parse = 0

    retry_user = {
        "role": "user",
        "content": (
            "Your previous reply was not valid JSON. Output exactly one JSON object "
            "matching the schema from the system message. No markdown fences, no prose."
        ),
    }

    for seed in tqdm(seed_rows, desc="Generating"):
        for _ in range(args.samples_per_seed):
            if running_cost >= args.budget_usd:
                budget_hit = True
                break
            task_type = random.choices(task_types, weights=weights, k=1)[0]
            messages = make_teacher_messages(task_type, seed, prompts_dir)

            sample = None
            last_raw = ""
            for attempt in range(2):
                resp = client.chat.completions.create(
                    model=args.teacher_model,
                    messages=messages,
                    temperature=0.7,
                    max_completion_tokens=args.max_output_tokens,
                )
                raw = resp.choices[0].message.content
                last_raw = (raw or "")[:500]
                usage = resp.usage
                cost = usage_cost_usd(usage, input_price, output_price)
                if not warned_missing_usage and (
                    usage is None
                    or (
                        (getattr(usage, "prompt_tokens", 0) or 0) == 0
                        and (getattr(usage, "completion_tokens", 0) or 0) == 0
                    )
                ):
                    print(
                        "\n[warn] API returned no token usage; spend shows $0.00. "
                        "Budget cap may not work until usage is present — check "
                        "https://platform.openai.com/usage for real cost.\n"
                    )
                    warned_missing_usage = True
                running_cost += cost

                try:
                    sample = parse_teacher_json(raw)
                    sample["metadata"] = sample.get("metadata") or {}
                    sample["metadata"]["cost_usd"] = cost
                    break
                except (ValueError, json.JSONDecodeError) as err:
                    if attempt == 0:
                        messages = list(messages) + [retry_user]
                        continue
                    skipped_parse += 1
                    tqdm.write(
                        f"[skip] JSON parse failed ({err}); preview: {last_raw!r}..."
                    )

            if sample is None:
                continue

            sample["id"] = sample.get("id") or f"ex_{uuid.uuid4().hex[:12]}"
            sample["task_type"] = sample.get("task_type", task_type)
            sample.setdefault("input", {})
            sample["input"].setdefault("question", seed.get("question", ""))
            sample["input"].setdefault("context", seed.get("context", ""))
            sample.setdefault("metadata", {})
            sample["metadata"]["source_model"] = args.teacher_model
            new_rows.append(sample)
        if budget_hit:
            break

    if args.append:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            for row in new_rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
    else:
        dump_jsonl(out_path, new_rows)

    total_n = prior_n + len(new_rows)
    print(f"Wrote {len(new_rows)} new samples -> {args.out_file} (total rows: {total_n})")
    if skipped_parse:
        print(f"Skipped {skipped_parse} responses (invalid JSON after retry).")
    print(f"Estimated spend (this run): ${running_cost:.2f} / ${args.budget_usd:.2f}")
    if new_rows and running_cost == 0.0:
        print(
            "(If that stayed $0.00, usage wasn’t in the response — "
            "confirm spend on the OpenAI dashboard.)"
        )
    if budget_hit:
        print("Stopped because budget was reached (this run).")


if __name__ == "__main__":
    main()
