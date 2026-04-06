# Miso Mini (Phi-4 Mini)

Train a small model to behave like a thinking system:

- draft
- critique
- refine
- optional debate
- optional compression

**Fast path (what we recommend):** with a venv, run `./scripts/run_best.sh` — it builds `data/wildchat_sft.jsonl` if missing, converts `data/miso_raw.jsonl` → `data/miso_sft.jsonl` if needed, then runs **curriculum** training (WildChat realism, then Miso self-refine on top). Final adapter: `outputs/phi4-stage2-miso`.

**Hardware:** [Unsloth](https://github.com/unslothai/unsloth) only supports **NVIDIA / AMD / Intel** GPUs. On **Apple Silicon (MPS)** or **CPU**, the same pipeline uses `scripts/train_lora_trl.py` (Transformers + PEFT + TRL, no 4-bit on MPS). For fastest training, use a Linux machine with an NVIDIA GPU and CUDA.

**Cloud GPU (RunPod, etc.):** `docs/RUNPOD.md` — pack with `scripts/pack_for_runpod.sh`, upload, run `scripts/runpod_install.sh`, then train with CUDA (Unsloth). Optional: `scripts/run_triad_on_runpod.sh` halts the machine **after a successful triad run** so idle GPU billing stops; otherwise **stop the pod** manually.

This project gives you a practical starter for:

1. synthetic data generation from a strong teacher model
2. dataset conversion to supervised fine-tuning (SFT) format
3. QLoRA training with Unsloth on `microsoft/Phi-4-mini-instruct`
4. evaluation of self-correction quality

## Project Layout

- `schemas/miso_schema_v1.json`: canonical JSON schema for your generated records
- `prompts/`: teacher prompts for refine/debate/compression samples
- `scripts/generate_synthetic.py`: generate JSONL data with OpenAI API
- `scripts/convert_to_sft.py`: convert canonical JSONL to chat SFT JSONL
- `scripts/train_unsloth.py`: train with Unsloth when **CUDA** is available (`--adapter-dir` for stage 2)
- `scripts/train_lora_trl.py`: same YAML/CLI, **Mac MPS / CPU** fallback (no Unsloth)
- `scripts/wildchat_to_sft.py`: WildChat → SFT JSONL (no API)
- `scripts/gsm8k_to_reasoning_sft.py`: GSM8K (parquet) → `data/reasoning_gsm8k.jsonl` for the reasoning adapter
- `scripts/merge_jsonl.py`: concatenate JSONL files (e.g. seed + GSM8K)
- `scripts/merge_sft_mix.py`: weighted mix + optional `[style:…]` tags
- **V2 balanced LoRA (fresh from base, no `--adapter-dir`):** `scripts/build_v2_mix.py` → `data/v2_mix_sft.jsonl`, `configs/train_v2_balanced.yaml`, `scripts/run_v2_train.sh` (40% WildChat / 30% GSM8K / 30% ARC+BBH, CoT + direct formats)
- `scripts/run_best.sh`: **one command** — prep data + `run_curriculum_train.sh` (start here)
- `scripts/run_curriculum_train.sh`: **stage 1** WildChat → **stage 2** Miso
- `scripts/run_weighted_mix_train.sh`: single run on 70/30-style mix
- `scripts/eval_pipeline.py`: quick metrics on raw teacher JSONL
- **Benchmarks:** `docs/BENCHMARK.md` — `pack_for_gpu_eval.sh` → GPU → `merge_adapter_for_eval.py` + [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) (`benchmark_suite.py`), `eval_triad_delta.py` + `plot_triad_delta.py` (triad figures)
- `configs/train_phi4mini_qlora.yaml`: default single-run train config
- `configs/train_stage1_wildchat.yaml` / `train_stage2_miso.yaml`: curriculum steps
- **Triad (optional):** `scripts/setup_triad_data.sh`, `train_triad_adapters.sh`, `infer_triad.py`, `configs/triad_*.yaml` — one base, **three separate LoRAs** (reasoning / response / critic)

## Triad architecture — 1 base, 3 focused LoRAs

Instead of one adapter that does everything, you train **three small heads** from the **same base** (`microsoft/Phi-4-mini-instruct` by default — swap in Phi-3 / Qwen2 / Gemma in the YAMLs if you want):

| Adapter | Role | Typical data |
|--------|------|----------------|
| **reasoning** | Step-by-step, logic, CoT | `data/reasoning_sft.jsonl` — seed in `reasoning_seed.jsonl`; add **GSM8K** via `scripts/gsm8k_to_reasoning_sft.py` → `reasoning_gsm8k.jsonl` (merged in `setup_triad_data.sh`) |
| **response** | Clear, human-sounding replies | WildChat → `data/response_sft.jsonl` with `[adapter:response]` |
| **critic** | Critique + improve (not generic Q&A) | Miso teacher JSON → `data/critic_sft.jsonl` with `[adapter:critic]` |

**Non-overlap:** keep datasets and prompts so the reasoning head doesn’t learn “sound pretty,” the response head doesn’t learn heavy CoT, and the critic doesn’t learn to answer without critiquing.

```bash
./scripts/setup_triad_data.sh
./scripts/train_triad_adapters.sh
python scripts/infer_triad.py --adapters-dir outputs/triad_adapters --question "Your question here"
```

Outputs: `outputs/triad_adapters/{reasoning,response,critic}/`. Same trainer selection as curriculum (Unsloth on CUDA, `train_lora_trl.py` on Mac).

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your API key for data generation:

```bash
export OPENAI_API_KEY=...
```

## Free SFT data: WildChat (no OpenAI API)

The [penfever/wildchat-50m](https://github.com/penfever/wildchat-50m) code loads HF datasets with a `conversation` column and runs **[vLLM](https://github.com/vllm-project/vllm)** to generate **new** assistant text with a **local** model. That avoids OpenAI billing but still needs a GPU and model weights.

To avoid both API cost and local generation, use **existing** user/assistant turns from **`allenai/WildChat-1M`** (see [dataset card](https://huggingface.co/datasets/allenai/WildChat-1M); ODC-BY — cite and comply). That gives standard instruction-tuning pairs, not the same JSON `draft/critique/improved` schema as teacher synthesis — you can blend both sources later.

```bash
pip install datasets tqdm
# Recommended on Python 3.14+: --parquet-only (Hub parquet shards; no datasets builder).
# Otherwise tries datasets first, then auto-falls back to parquet on failure.
python scripts/wildchat_to_sft.py --parquet-only --max-rows 10000 --output data/wildchat_sft.jsonl
# optional: --mode all_turns | --skip-toxic | --streaming (omit --parquet-only)
```

Train on `data/wildchat_sft.jsonl` the same way as `data/miso_sft.jsonl`, or use the mix / curriculum flow below.

## Mix WildChat + teacher data and train

**Behavior distribution:** WildChat teaches *how people actually prompt and how a helpful assistant sounds*; Miso SFT teaches *structured self-refinement* (JSON targets). Combining them beats using either alone for a small model.

### Option B — Curriculum (recommended)

```bash
source .venv/bin/activate
chmod +x scripts/run_best.sh scripts/run_curriculum_train.sh
./scripts/run_best.sh
```

Optional: `WILDCHAT_MAX_ROWS=5000` to use a smaller WildChat slice. Or build data yourself, then:

```bash
./scripts/run_curriculum_train.sh
```

Stage 1 saves LoRA to `outputs/phi4-stage1-wildchat`. Stage 2 loads that adapter and continues on `data/miso_sft.jsonl` into `outputs/phi4-stage2-miso` (final adapter).

Override paths if needed: `WC_SFT=... MISO_SFT=... STAGE1_OUT=... STAGE2_OUT=... ./scripts/run_curriculum_train.sh`.

### Option A — Weighted single run (~70% WildChat / 30% Miso)

```bash
WILDCHAT_RATIO=0.7 MIX_TARGET_SIZE=8000 ./scripts/run_weighted_mix_train.sh
```

Set `TAG_STYLE=1` to prefix system prompts with `[style:casual_chat]` vs `[style:reasoning]` (light conditioning).

Or merge only:

```bash
python scripts/merge_sft_mix.py \
  --wildchat data/wildchat_sft.jsonl \
  --miso data/miso_sft.jsonl \
  --wildchat-ratio 0.7 \
  --target-size 8000 \
  --output data/mixed_sft.jsonl \
  --tag-style
python scripts/train_unsloth.py \
  --config configs/train_phi4mini_qlora.yaml \
  --train-file data/mixed_sft.jsonl \
  --output-dir outputs/phi4-mixed-lora
```

### Option C — tags only

Use `--tag-style` on `merge_sft_mix.py` (or `TAG_STYLE=1` with the mix train script) so the model sees discrete “modes” in the system line.

## Make it happen (grow a real dataset)

1. Put your key in `openai_key.txt` (one line) or set `OPENAI_API_KEY`.
2. Build seeds and generate in budgeted batches (append mode keeps growing `data/miso_raw.jsonl`):

```bash
source .venv/bin/activate   # optional
export BUDGET_USD=3.5       # per run; re-run until you hit your OpenAI cap
./scripts/run_make_data.sh
```

Re-run `./scripts/run_make_data.sh` whenever you add budget — it **appends** new rows and rebuilds `data/miso_sft.jsonl`.

- **83 diverse seeds** live in `data/seeds_bulk.jsonl` (from `scripts/bootstrap_seeds.py`). Edit or re-run bootstrap after changing the script.
- Aim for **hundreds to thousands** of rows before expecting a clear training effect.

## 2) Create seed tasks

Make a seed file with one JSON object per line:

```json
{"question":"What is overfitting in machine learning?","context":""}
{"question":"Write Python code to reverse a linked list.","context":""}
```

Save as `data/seeds.jsonl`.

## 3) Generate synthetic training data

```bash
python scripts/generate_synthetic.py \
  --seed-file data/seeds.jsonl \
  --out-file data/miso_raw.jsonl \
  --teacher-model gpt-5 \
  --samples-per-seed 2
```

## 4) Convert to SFT format

```bash
python scripts/convert_to_sft.py \
  --input data/miso_raw.jsonl \
  --output data/miso_sft.jsonl \
  --mode train
```

## 5) Train (Unsloth + QLoRA)

```bash
python scripts/train_unsloth.py \
  --config configs/train_phi4mini_qlora.yaml \
  --train-file data/miso_sft.jsonl \
  --output-dir outputs/phi4mini-miso-lora
```

## 6) Evaluate

```bash
python scripts/eval_pipeline.py \
  --eval-file data/miso_raw.jsonl \
  --pred-file outputs/eval_predictions.jsonl
```

## Notes

- Start small (1k-5k samples), inspect quality, then scale.
- Keep outputs concise; overlong reasoning hurts compact models.
- Use curriculum mixing:
  - 40% refine
  - 30% direct answer
  - 20% compression
  - 10% debate
