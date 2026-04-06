#!/usr/bin/env python3
"""
Run the standard lm-eval benchmark matrix (mixed few-shot settings):

  - GSM8K (0-shot only; 8-shot dropped — fewer-shot interference)
  - HellaSwag (5-shot)
  - ARC-Challenge (10-shot)
  - MMLU mini (5-shot): 8 subjects — math, physics, CS, commonsense, history
    (group `mmlu_mini_subset` in configs/mmlu_mini_subset.yaml)
  - BigBench Hard (0-shot, chain-of-thought): task group bbh_cot_zeroshot

Each benchmark is a separate `lm_eval run` so per-task shot counts are correct.

  pip install -r requirements-eval.txt

  python scripts/benchmark_suite.py outputs/merged_reasoning
  python scripts/benchmark_suite.py outputs/merged_reasoning --limit 10   # smoke test
  python scripts/benchmark_suite.py MODEL --only gsm8k_0shot,arc_challenge_10shot,hellaswag_0shot
  python scripts/benchmark_suite.py --peft outputs/lora_v2_balanced --only gsm8k_0shot,...  # no merged copy (saves disk)

Legacy single run (all 0-shot, gsm8k + mmlu + arc_challenge):

  python scripts/benchmark_suite.py --legacy outputs/merged_reasoning
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# (lm-eval task or group, num_fewshot, output dir stem, use_repo_configs_include_path)
# When True, passes --include_path <repo>/configs so custom groups (e.g. mmlu_mini_subset) resolve.
DEFAULT_RUNS: tuple[tuple[str, int, str, bool], ...] = (
    ("gsm8k", 0, "gsm8k_0shot", False),
    ("hellaswag", 5, "hellaswag_5shot", False),
    ("arc_challenge", 10, "arc_challenge_10shot", False),
    ("mmlu_mini_subset", 5, "mmlu_mini_5shot", True),
    ("bbh_cot_zeroshot", 0, "bbh_cot_zeroshot", False),
)

# Extra presets (use with --only), e.g. quick HellaSwag 0-shot + read acc_norm in results JSON.
EXTRA_RUNS: tuple[tuple[str, int, str, bool], ...] = (
    ("hellaswag", 0, "hellaswag_0shot", False),
)

ALL_RUNS: tuple[tuple[str, int, str, bool], ...] = DEFAULT_RUNS + EXTRA_RUNS


def _repo_configs_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "configs"

LEGACY_TASKS = "gsm8k,mmlu,arc_challenge"


def _lm_eval_argv() -> list[str]:
    """Same interpreter as this script so venv works when `lm_eval` is not on PATH."""
    return [sys.executable, "-m", "lm_eval"]


def _run_one(
    *,
    model_args: str,
    task: str,
    num_fewshot: int,
    run_output_dir: Path,
    device: str,
    limit: int | None,
    include_repo_configs: bool = False,
) -> int:
    # lm-eval renames *.json paths to stem_<timestamp>.json; use a directory instead.
    run_output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        *_lm_eval_argv(),
        "run",
    ]
    if include_repo_configs:
        cfg = _repo_configs_dir()
        if not cfg.is_dir():
            print(f"Missing configs directory (needed for task group {task}): {cfg}", file=sys.stderr)
            return 1
        cmd.extend(["--include_path", str(cfg)])
    cmd.extend(
        [
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        task,
        "--num_fewshot",
        str(num_fewshot),
        "--device",
        device,
        "--batch_size",
        "auto",
        "--output_path",
        str(run_output_dir.resolve()),
    ]
    )
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    print("command:", " ".join(cmd))
    print()
    # Propagate unbuffered child logs when stdout is a pipe or file (e.g. nohup).
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    return subprocess.run(cmd, env=env).returncode


def _legacy_main(args: argparse.Namespace) -> int:
    if args.peft:
        peft_path = Path(args.peft)
        if not peft_path.is_dir():
            print(f"PEFT adapter directory not found: {peft_path}", file=sys.stderr)
            sys.exit(1)
        model_args = f"pretrained={args.base},peft={peft_path.resolve()}"
    else:
        if not args.model:
            print("Legacy mode requires a merged model path (or use --peft).", file=sys.stderr)
            sys.exit(1)
        model = Path(args.model)
        if not model.is_dir():
            print(f"Model directory not found: {model}", file=sys.stderr)
            sys.exit(1)
        model_args = f"pretrained={model.resolve()}"

    out = args.output
    if not out:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = f"eval/benchmark_suite_legacy_0shot_{ts}.json"
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        *_lm_eval_argv(),
        "run",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        args.tasks,
        "--num_fewshot",
        "0",
        "--device",
        args.device,
        "--batch_size",
        "auto",
        "--output_path",
        str(out_path.resolve()),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    print("command:", " ".join(cmd))
    print("fewshot: 0 (all tasks in --tasks)")
    print("output:", out_path)
    print()

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        return r.returncode
    print(f"Wrote {out_path}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="lm-eval benchmark matrix (GSM8K 0-shot, HellaSwag, ARC, MMLU-mini, BBH-CoT) or --legacy 0-shot trio.",
    )
    p.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Path to merged HF model directory (omit if using --peft)",
    )
    p.add_argument(
        "--peft",
        default=None,
        metavar="ADAPTER_DIR",
        help="PEFT adapter directory; lm-eval loads base (--base) + adapter (no merged folder on disk)",
    )
    p.add_argument(
        "--base",
        default="microsoft/Phi-4-mini-instruct",
        help="With --peft: Hugging Face model id or path for the base weights",
    )
    p.add_argument(
        "--legacy",
        action="store_true",
        help=f"Single run: --tasks {LEGACY_TASKS} with --num_fewshot 0 (old behavior)",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Legacy: single JSON path. Default: eval/benchmark_suite_legacy_0shot_<ts>.json",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Mixed suite: directory for per-run JSON files (default: eval/benchmark_suite_<timestamp>/)",
    )
    p.add_argument("--device", default="cuda", help="cuda | cuda:0 | cpu | mps")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit examples per task (smoke test; not for publishable numbers)",
    )
    p.add_argument(
        "--only",
        default=None,
        metavar="STEMS",
        help="Comma-separated run ids (stems), e.g. gsm8k_0shot,arc_challenge_10shot,hellaswag_0shot",
    )
    p.add_argument(
        "--tasks",
        default=LEGACY_TASKS,
        help=f"Legacy only: comma-separated lm-eval tasks (default: {LEGACY_TASKS})",
    )
    args = p.parse_args()

    if args.legacy:
        sys.exit(_legacy_main(args))

    if args.peft:
        peft_path = Path(args.peft)
        if not peft_path.is_dir():
            print(f"PEFT adapter directory not found: {peft_path}", file=sys.stderr)
            sys.exit(1)
        model_args_value = f"pretrained={args.base},peft={peft_path.resolve()}"
        manifest_model = model_args_value
    else:
        model_path = args.model or "outputs/merged_reasoning"
        model = Path(model_path)
        if not model.is_dir():
            print(f"Model directory not found: {model}", file=sys.stderr)
            print("Merge an adapter first, e.g.:", file=sys.stderr)
            print(
                "  python scripts/merge_adapter_for_eval.py "
                "--adapter outputs/triad_adapters/reasoning --output outputs/merged_reasoning",
                file=sys.stderr,
            )
            print("Or run with --peft ADAPTER_DIR (base + LoRA, no full merge on disk).", file=sys.stderr)
            sys.exit(1)
        model_args_value = f"pretrained={model.resolve()}"
        manifest_model = str(model.resolve())

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"eval/benchmark_suite_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        by_stem = {r[2]: r for r in ALL_RUNS}
        unknown = [s for s in wanted if s not in by_stem]
        if unknown:
            print(f"Unknown --only stem(s): {unknown}. Known: {sorted(by_stem)}", file=sys.stderr)
            sys.exit(1)
        runs_to_do = [by_stem[s] for s in wanted]
    else:
        runs_to_do = list(DEFAULT_RUNS)

    manifest: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": manifest_model,
        "runs": [],
    }

    print("Benchmark matrix (separate lm_eval runs):")
    for task, nshot, stem, need_cfg in runs_to_do:
        extra = f" --include_path {_repo_configs_dir()}" if need_cfg else ""
        print(f"  - {stem}:{extra} --tasks {task} --num_fewshot {nshot}")
    print("output_dir:", out_dir.resolve())
    print()

    for task, nshot, stem, need_cfg in runs_to_do:
        run_output_dir = out_dir / stem
        rc = _run_one(
            model_args=model_args_value,
            task=task,
            num_fewshot=nshot,
            run_output_dir=run_output_dir,
            device=args.device,
            limit=args.limit,
            include_repo_configs=need_cfg,
        )
        entry = {
            "id": stem,
            "tasks": task,
            "num_fewshot": nshot,
            "include_repo_configs": need_cfg,
            "output_dir": str(run_output_dir.resolve()),
            "exit_code": rc,
        }
        manifest["runs"].append(entry)
        if rc != 0:
            print(f"Run failed ({stem}), exit {rc}", file=sys.stderr)
            manifest_path = out_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            sys.exit(rc)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(runs_to_do)} lm-eval output dirs under {out_dir} (see each subdir for results_*.json)")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
