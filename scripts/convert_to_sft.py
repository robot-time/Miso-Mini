import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_user_prompt(row):
    q = row["input"].get("question", "")
    c = row["input"].get("context", "")
    if c:
        return f"Question:\n{q}\n\nContext:\n{c}"
    return f"Question:\n{q}"


def build_assistant_target(row):
    task = row.get("task_type", "qa_refine")
    t = row.get("targets", {})

    if task == "qa_refine":
        critique = t.get("critique", [])
        return json.dumps(
            {
                "draft": t.get("draft", ""),
                "critique": critique,
                "improved": t.get("improved", ""),
            },
            ensure_ascii=True,
        )
    if task == "debate":
        return json.dumps(
            {
                "perspectives": t.get("perspectives", {}),
                "adjudication": t.get("adjudication", ""),
                "improved": t.get("improved", ""),
            },
            ensure_ascii=True,
        )
    if task == "compression":
        return json.dumps(
            {
                "compressed_memory": t.get("compressed_memory", ""),
                "answer": t.get("answer", ""),
            },
            ensure_ascii=True,
        )
    return json.dumps(t, ensure_ascii=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="train", choices=["train", "eval"])
    parser.add_argument(
        "--system-prefix",
        default="",
        help="Optional text prepended to the system message (e.g. triad critic tag).",
    )
    args = parser.parse_args()

    base_system = (
        "You are a concise reasoning assistant. Think in structured steps and "
        "provide a refined final response."
    )
    if args.system_prefix:
        base_system = args.system_prefix.strip() + " " + base_system

    raw = load_jsonl(Path(args.input))
    out = []
    for row in raw:
        user_prompt = build_user_prompt(row)
        assistant = build_assistant_target(row)
        out.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": base_system,
                    },
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant},
                ]
            }
        )

    dump_jsonl(Path(args.output), out)
    print(f"Wrote {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
