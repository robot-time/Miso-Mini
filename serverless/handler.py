#!/usr/bin/env python3
"""
RunPod Serverless worker: triad pipeline (reasoning → response → critic).

Environment:
  TRIAD_ADAPTERS_DIR  — directory containing reasoning/, response/, critic/ LoRA folders (default /app/adapters)
  TRIAD_CONFIG        — path to YAML with model_name + load_in_4bit (default /app/configs/triad_reasoning.yaml)
  HF_HOME             — optional; set to a persistent volume path for faster cold starts

Input JSON (job["input"]):
  question | prompt | message  — required user text
  max_new_tokens                 — optional, default 384

Output JSON:
  user, reasoning, draft, final
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

# /app/infer_triad.py (see Dockerfile)
sys.path.insert(0, "/app")

import runpod  # noqa: E402

import infer_triad  # noqa: E402

log = logging.getLogger("triad_serverless")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_runtime_lock = threading.Lock()
_runtime: infer_triad.TriadHotSwapRuntime | None = None


def _adapters_dir() -> Path:
    return Path(os.environ.get("TRIAD_ADAPTERS_DIR", "/app/adapters"))


def _config_path() -> Path:
    return Path(os.environ.get("TRIAD_CONFIG", "/app/configs/triad_reasoning.yaml"))


def get_runtime() -> infer_triad.TriadHotSwapRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            cfg = infer_triad.load_yaml_config(_config_path())
            base = _adapters_dir()
            r, s, c = base / "reasoning", base / "response", base / "critic"
            log.info("Loading triad (single base + 3 adapters) from %s", base)
            _runtime = infer_triad.TriadHotSwapRuntime(cfg, r, s, c)
            log.info("Triad ready.")
        return _runtime


def handler(job):
    try:
        job_input = job.get("input") if isinstance(job, dict) else {}
        if not isinstance(job_input, dict):
            return {"error": "input must be a JSON object"}

        text = (
            job_input.get("question")
            or job_input.get("prompt")
            or job_input.get("message")
            or ""
        )
        text = str(text).strip()
        if not text:
            return {"error": "missing question|prompt|message in input"}

        max_new = int(job_input.get("max_new_tokens", 384))
        max_new = max(16, min(max_new, 4096))

        rt = get_runtime()
        out = rt.run(text, max_new_tokens=max_new)
        return out
    except Exception as e:
        log.exception("handler error")
        return {"error": str(e), "type": type(e).__name__}


runpod.serverless.start({"handler": handler})
