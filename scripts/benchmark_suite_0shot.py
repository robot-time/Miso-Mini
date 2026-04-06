#!/usr/bin/env python3
"""Backward-compatible entry: old 0-shot gsm8k + mmlu + arc_challenge in one lm_eval run."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    suite = here / "benchmark_suite.py"
    cmd = [sys.executable, str(suite), "--legacy", *sys.argv[1:]]
    raise SystemExit(subprocess.run(cmd).returncode)
