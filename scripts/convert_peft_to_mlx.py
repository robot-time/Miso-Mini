#!/usr/bin/env python3
"""
Convert PEFT/HuggingFace LoRA adapters to MLX-LM format.

Usage:
  python scripts/convert_peft_to_mlx.py          # converts all three triad adapters
  python scripts/convert_peft_to_mlx.py --adapter reasoning  # one adapter only
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def convert_adapter(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)

    peft_cfg_path = src / "adapter_config.json"
    peft_cfg = json.loads(peft_cfg_path.read_text())

    rank = peft_cfg.get("r", 32)
    alpha = peft_cfg.get("lora_alpha", rank)
    dropout = peft_cfg.get("lora_dropout", 0.0)
    scale = alpha / rank

    weights_src = src / "adapter_model.safetensors"
    mlx_weights: dict[str, np.ndarray] = {}

    with safe_open(str(weights_src), framework="numpy") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # Strip PEFT prefix: base_model.model.model.layers... → model.layers...
            new_key = key.replace("base_model.model.", "", 1)
            # Rename lora_A/B to mlx convention
            if ".lora_A.weight" in key:
                new_key = new_key.replace(".lora_A.weight", ".lora_a")
                # PEFT: [rank, in_features] → MLX: [in_features, rank]
                tensor = tensor.T
            elif ".lora_B.weight" in key:
                new_key = new_key.replace(".lora_B.weight", ".lora_b")
                # PEFT: [out_features, rank] → MLX: [rank, out_features]
                tensor = tensor.T
            mlx_weights[new_key] = tensor.astype(np.float16)

    save_file(mlx_weights, str(dst / "adapters.safetensors"))

    # Find how many layers have adapters
    num_layers = len(set(
        int(k.split("layers.")[1].split(".")[0])
        for k in mlx_weights
        if "layers." in k
    ))

    mlx_cfg = {
        "lora_layers": num_layers,
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": rank,
            "alpha": float(alpha),
            "scale": scale,
            "dropout": dropout,
        },
    }
    (dst / "adapter_config.json").write_text(json.dumps(mlx_cfg, indent=2))

    # Copy tokenizer files if present (mlx-lm may need them)
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        if (src / fname).exists():
            shutil.copy2(src / fname, dst / fname)

    print(f"  {src.name} → {dst} ({len(mlx_weights)} tensors, {num_layers} layers, scale={scale:.3f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapters-dir", default="outputs/triad_adapters")
    p.add_argument("--out-dir", default="outputs/triad_adapters_mlx")
    p.add_argument("--adapter", default=None, help="reasoning | response | critic (all if omitted)")
    args = p.parse_args()

    src_base = Path(args.adapters_dir)
    dst_base = Path(args.out_dir)

    names = [args.adapter] if args.adapter else ["reasoning", "response", "critic"]
    print(f"Converting {len(names)} adapter(s) from PEFT → MLX format...")
    for name in names:
        convert_adapter(src_base / name, dst_base / name)
    print("Done.")


if __name__ == "__main__":
    main()
