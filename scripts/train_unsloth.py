# Unsloth must load before transformers / trl / peft (see unsloth docs).
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import argparse
import json
import os
from pathlib import Path

import yaml
from datasets import Dataset
from peft import PeftModel
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Existing LoRA folder from a previous stage (curriculum stage 2). "
        "Skips fresh get_peft_model and continues training that adapter.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = load_jsonl(Path(args.train_file))
    ds = Dataset.from_list(rows).map(format_chat)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=cfg["dtype"],
        load_in_4bit=cfg["load_in_4bit"],
    )

    adapter_path = Path(args.adapter_dir) if args.adapter_dir else None
    if adapter_path is not None:
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"adapter-dir not found: {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=True,
        )
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora"]["r"],
            target_modules=cfg["lora"]["target_modules"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            bias=cfg["lora"]["bias"],
            use_gradient_checkpointing="unsloth",
        )

    # TRL defaults to many workers for dataset.map(); Unsloth tokenizers + multiprocessing often fail to pickle
    # (ConfigModuleInstance, NoneType iteration). Default 1 process; override with DATASET_NUM_PROC if needed.
    _ds_proc = max(1, int(os.environ.get("DATASET_NUM_PROC", "1")))

    train_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        warmup_steps=cfg["training"]["warmup_steps"],
        max_steps=cfg["training"]["max_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        optim=cfg["training"]["optim"],
        weight_decay=cfg["training"]["weight_decay"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        seed=cfg["training"]["seed"],
        fp16=cfg["training"]["fp16"],
        bf16=cfg["training"]["bf16"],
        report_to=cfg["training"]["report_to"],
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

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training finished -> {args.output_dir}")


if __name__ == "__main__":
    main()
