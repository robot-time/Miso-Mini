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


def score_refine_sample(row):
    t = row.get("targets", {})
    draft = t.get("draft", "")
    improved = t.get("improved", "")
    critique = t.get("critique", [])
    changed = int(draft.strip() != improved.strip())
    critique_nonempty = int(isinstance(critique, list) and len(critique) > 0)
    return {
        "changed_answer": changed,
        "has_critique": critique_nonempty,
        "self_correction_score": 0.5 * changed + 0.5 * critique_nonempty,
    }


def score_compression_sample(row):
    t = row.get("targets", {})
    ctx = row.get("input", {}).get("context", "")
    mem = t.get("compressed_memory", "")
    ans = t.get("answer", "")
    compression_ratio = (len(mem) / max(len(ctx), 1)) if ctx else 1.0
    valid = int(bool(mem.strip()) and bool(ans.strip()))
    return {
        "compression_ratio": compression_ratio,
        "compression_valid": valid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--pred-file", required=True)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.eval_file))
    scored = []
    agg = {
        "count": 0,
        "refine_count": 0,
        "compression_count": 0,
        "avg_self_correction_score": 0.0,
        "avg_compression_ratio": 0.0,
    }

    sc_sum = 0.0
    cr_sum = 0.0
    for row in rows:
        task = row.get("task_type")
        score = {}
        if task == "qa_refine":
            score = score_refine_sample(row)
            agg["refine_count"] += 1
            sc_sum += score["self_correction_score"]
        elif task == "compression":
            score = score_compression_sample(row)
            agg["compression_count"] += 1
            cr_sum += score["compression_ratio"]
        scored.append({"id": row.get("id"), "task_type": task, "score": score})
        agg["count"] += 1

    if agg["refine_count"] > 0:
        agg["avg_self_correction_score"] = sc_sum / agg["refine_count"]
    if agg["compression_count"] > 0:
        agg["avg_compression_ratio"] = cr_sum / agg["compression_count"]

    dump_jsonl(Path(args.pred_file), scored)
    print(json.dumps(agg, ensure_ascii=True, indent=2))
    print(f"Scored rows -> {args.pred_file}")


if __name__ == "__main__":
    main()
