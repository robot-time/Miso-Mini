#!/usr/bin/env python3
"""Concatenate JSONL files in order (one JSON object per line). Empty lines skipped."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path, help="Input JSONL files (order preserved)")
    p.add_argument("--output", "-o", required=True, type=Path)
    args = p.parse_args()

    total = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for path in args.inputs:
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    json.loads(line)  # validate
                    out.write(line + "\n")
                    total += 1
    print(f"Merged {len(args.inputs)} files -> {args.output} ({total} lines)")


if __name__ == "__main__":
    main()
