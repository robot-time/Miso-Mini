"""
LoRA SFT without Unsloth — use on Apple Silicon (MPS) or CPU where Unsloth is unavailable.
Unsloth only supports NVIDIA / AMD / Intel GPUs.

Same CLI and YAML as train_unsloth.py. CUDA: 4-bit QLoRA when cfg.load_in_4bit is true.
MPS/CPU: float16/float32, no bitsandbytes 4-bit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_chat(row):
    text = ""
    for m in row["messages"]:
        role = m["role"].upper()
        text += f"<|{role}|>\n{m['content']}\n"
    return {"text": text}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-dir", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = load_jsonl(Path(args.train_file))
    ds = Dataset.from_list(rows).map(format_chat)

    device = pick_device()
    use_4bit = bool(cfg.get("load_in_4bit")) and device == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)

    adapter_path = Path(args.adapter_dir) if args.adapter_dir else None
    if adapter_path is not None:
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"adapter-dir not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    else:
        lc = cfg["lora"]
        peft_config = LoraConfig(
            r=lc["r"],
            lora_alpha=lc["alpha"],
            target_modules=lc["target_modules"],
            lora_dropout=lc["dropout"],
            bias=lc.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)

    model.gradient_checkpointing_enable()

    optim = cfg["training"]["optim"]
    if device != "cuda" and "8bit" in optim:
        optim = "adamw_torch"

    per_device_bs = cfg["training"]["per_device_train_batch_size"]
    if device == "mps":
        per_device_bs = min(int(per_device_bs), 1)

    use_fp16 = (cfg["training"]["fp16"] and device == "cuda") or device == "mps"
    use_bf16 = bool(cfg["training"]["bf16"]) and device == "cuda"

    # Avoid datasets multiprocessing pickling issues (esp. with some tokenizers). Default 1; see DATASET_NUM_PROC.
    _ds_proc = max(1, int(os.environ.get("DATASET_NUM_PROC", "1")))

    train_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        warmup_steps=cfg["training"]["warmup_steps"],
        max_steps=cfg["training"]["max_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        optim=optim,
        weight_decay=cfg["training"]["weight_decay"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        seed=cfg["training"]["seed"],
        fp16=use_fp16,
        bf16=use_bf16,
        report_to=cfg["training"]["report_to"],
        gradient_checkpointing=True,
        dataset_text_field="text",
        max_length=cfg["max_seq_length"],
        dataset_num_proc=_ds_proc,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=train_args,
    )

    print(f"[train_lora_trl] device={device} use_4bit={use_4bit} optim={optim}")
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training finished -> {args.output_dir}")


if __name__ == "__main__":
    main()
