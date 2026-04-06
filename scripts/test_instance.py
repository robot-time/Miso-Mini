import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(question: str, context: str) -> str:
    if context.strip():
        return (
            "<|SYSTEM|>\n"
            "You are a concise reasoning assistant.\n"
            "<|USER|>\n"
            f"Question:\n{question}\n\nContext:\n{context}\n"
            "<|ASSISTANT|>\n"
        )
    return (
        "<|SYSTEM|>\n"
        "You are a concise reasoning assistant.\n"
        "<|USER|>\n"
        f"Question:\n{question}\n"
        "<|ASSISTANT|>\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="outputs/phi4mini-miso-lora",
        help="Path to trained adapter/model directory. If missing, uses base model.",
    )
    parser.add_argument(
        "--base-model",
        default="microsoft/Phi-4-mini-instruct",
        help="Fallback base model when model-dir does not exist.",
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    model_name = args.model_dir if os.path.exists(args.model_dir) else args.base_model
    if model_name == args.base_model:
        print(f"[info] Adapter not found at {args.model_dir}. Using base model.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )

    prompt = build_prompt(args.question, args.context)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print("\n=== RAW OUTPUT ===\n")
    print(text)


if __name__ == "__main__":
    main()
