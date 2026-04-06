#!/usr/bin/env python3
"""
Every user message goes through all three models, in order — no shortcuts:

  reasoning → response → critic

Each stage loads base + one LoRA (fresh base load per stage; HF cache helps).

Override PROMPT_* in this file to match your training tags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Role isolation: each head has a narrow job (adjust to match your training prompts).
PROMPT_REASONING = (
    "[adapter:reasoning]\nUser question:\n{user}\n"
    "Think step by step. Show reasoning, then one short conclusion line."
)
PROMPT_RESPONSE = (
    "[adapter:response]\nUser question:\n{user}\n"
    "Reasoning draft (do not repeat long reasoning; polish into a clear helpful reply):\n{reasoning}\n"
    "Write the final user-facing reply only. Be concise and natural."
)
PROMPT_CRITIC = (
    "[adapter:critic]\nUser question:\n{user}\n"
    "Draft reply:\n{draft}\n"
    "Critique briefly, then output ONLY the improved final reply."
)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_yaml_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_base_and_tokenizer(cfg: dict):
    """Load tokenizer + base causal LM once (no PEFT). Used by hot-swap and per-adapter loads."""
    device = pick_device()
    use_4bit = bool(cfg.get("load_in_4bit")) and device == "cuda"
    mid = cfg["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(mid)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            mid,
            quantization_config=bnb,
            device_map="auto",
        )
    else:
        # CPU/MPS: load directly in float16 on CPU — avoids the double-buffer
        # OOM that happens when calling .to("mps") on a 7GB model on 16GB RAM.
        # Apple Silicon unified memory runs CPU inference fast enough for testing.
        model = AutoModelForCausalLM.from_pretrained(
            mid,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        if device == "mps":
            model = model.to("mps")
        elif device == "cuda":
            model = model.to("cuda")

    return model, tokenizer, device


def load_model_tokenizer(cfg: dict, adapter_dir: Path):
    model, tokenizer, device = build_base_and_tokenizer(cfg)
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    return model, tokenizer, device


def generate(model, tokenizer, device: str, prompt: str, max_new: int = 512) -> str:
    dev = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.35,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt) :].strip()
    return text.strip()


def run_triad(
    cfg: dict,
    r_dir: Path,
    s_dir: Path,
    c_dir: Path,
    user: str,
    max_new_tokens: int,
    *,
    show_stages: bool,
) -> dict:
    """Always runs all three stages. Returns dict with reasoning, draft, final."""
    user = user.strip()
    if not user:
        raise ValueError("empty message")

    if show_stages:
        print("--- Stage 1: reasoning ---", file=sys.stderr)
    m1, t1, dev = load_model_tokenizer(cfg, r_dir)
    p1 = PROMPT_REASONING.format(user=user)
    reasoning = generate(m1, t1, dev, p1, max_new=max_new_tokens)
    del m1
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if show_stages:
        print(reasoning[:800] + ("..." if len(reasoning) > 800 else ""), file=sys.stderr)
        print(file=sys.stderr)

    if show_stages:
        print("--- Stage 2: response ---", file=sys.stderr)
    m2, t2, dev = load_model_tokenizer(cfg, s_dir)
    p2 = PROMPT_RESPONSE.format(user=user, reasoning=reasoning)
    draft = generate(m2, t2, dev, p2, max_new=max_new_tokens)
    del m2
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if show_stages:
        print(draft[:800] + ("..." if len(draft) > 800 else ""), file=sys.stderr)
        print(file=sys.stderr)

    if show_stages:
        print("--- Stage 3: critic ---", file=sys.stderr)
    m3, t3, dev = load_model_tokenizer(cfg, c_dir)
    p3 = PROMPT_CRITIC.format(user=user, draft=draft)
    final = generate(m3, t3, dev, p3, max_new=max_new_tokens)
    del m3

    return {
        "user": user,
        "reasoning": reasoning,
        "draft": draft,
        "final": final,
    }


class TriadHotSwapRuntime:
    """
    One base load + three LoRA adapters (PEFT hot-swap). Much faster than loading the base three times
    per request — intended for RunPod Serverless and similar always-on GPU workers.
    """

    def __init__(
        self,
        cfg: dict,
        r_dir: Path,
        s_dir: Path,
        c_dir: Path,
    ):
        for d, name in ((r_dir, "reasoning"), (s_dir, "response"), (c_dir, "critic")):
            if not d.is_dir():
                raise FileNotFoundError(f"Missing adapter dir ({name}): {d}")

        base, self.tokenizer, self.device = build_base_and_tokenizer(cfg)
        self.model = PeftModel.from_pretrained(
            base,
            str(r_dir),
            adapter_name="reasoning",
            is_trainable=False,
        )
        self.model.load_adapter(str(s_dir), adapter_name="response")
        self.model.load_adapter(str(c_dir), adapter_name="critic")
        self.model.eval()

    def run(self, user: str, max_new_tokens: int = 384) -> dict:
        user = user.strip()
        if not user:
            raise ValueError("empty message")

        self.model.set_adapter("reasoning")
        reasoning = generate(
            self.model,
            self.tokenizer,
            self.device,
            PROMPT_REASONING.format(user=user),
            max_new_tokens,
        )

        self.model.set_adapter("response")
        draft = generate(
            self.model,
            self.tokenizer,
            self.device,
            PROMPT_RESPONSE.format(user=user, reasoning=reasoning),
            max_new_tokens,
        )

        self.model.set_adapter("critic")
        final = generate(
            self.model,
            self.tokenizer,
            self.device,
            PROMPT_CRITIC.format(user=user, draft=draft),
            max_new_tokens,
        )

        return {
            "user": user,
            "reasoning": reasoning,
            "draft": draft,
            "final": final,
        }


def run_triad_hot_swap(
    cfg: dict,
    r_dir: Path,
    s_dir: Path,
    c_dir: Path,
    user: str,
    max_new_tokens: int,
    *,
    show_stages: bool,
) -> dict:
    """One-shot convenience: build runtime, run one message, discard (for testing)."""
    if show_stages:
        print("--- Hot-swap triad (single base + 3 adapters) ---", file=sys.stderr)
    rt = TriadHotSwapRuntime(cfg, r_dir, s_dir, c_dir)
    return rt.run(user, max_new_tokens)


def main():
    p = argparse.ArgumentParser(
        description="Run every message through reasoning → response → critic (all three, always).",
    )
    p.add_argument("--adapters-dir", default="outputs/triad_adapters", help="Parent dir with reasoning/, response/, critic/")
    p.add_argument("--config", default="configs/triad_reasoning.yaml", help="YAML for model_name + quant (shared across adapters)")
    p.add_argument("--question", default="", help="Single user message (omit with --stdin)")
    p.add_argument("--stdin", action="store_true", help="Read one message per line; each line runs the full 3-stage pipeline")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON object per message (reasoning, draft, final); stage text to stderr unless --quiet",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="With --json: stderr is silent; only JSON lines on stdout",
    )
    p.add_argument(
        "--hot-swap",
        action="store_true",
        help="Load base model once and swap LoRA adapters (faster; same outputs in spirit as default)",
    )
    args = p.parse_args()

    cfg = load_yaml_config(Path(args.config))
    base_dir = Path(args.adapters_dir)
    r_dir = base_dir / "reasoning"
    s_dir = base_dir / "response"
    c_dir = base_dir / "critic"
    for d in (r_dir, s_dir, c_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Missing adapter dir: {d}")

    if args.json:
        show_stages = not args.quiet
    else:
        show_stages = True

    def handle_one(user_line: str) -> None:
        if args.hot_swap:
            out = run_triad_hot_swap(
                cfg,
                r_dir,
                s_dir,
                c_dir,
                user_line,
                args.max_new_tokens,
                show_stages=show_stages,
            )
        else:
            out = run_triad(
                cfg,
                r_dir,
                s_dir,
                c_dir,
                user_line,
                args.max_new_tokens,
                show_stages=show_stages,
            )
        if args.json:
            print(json.dumps(out, ensure_ascii=True))
        else:
            print("=== Final ===")
            print(out["final"])

    if args.stdin:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            handle_one(line)
        return

    if not args.question.strip():
        p.error("Pass --question \"...\" or --stdin")
    handle_one(args.question)


if __name__ == "__main__":
    main()
