# Benchmarking Miso-Mini (base + LoRAs / triad)

You have **two** different things to measure:

1. **Standard tasks** (GSM8K, HellaSwag, ARC-Challenge, BBH, etc.) — [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), with **per-benchmark shot counts** (see below).
2. **Triad-specific behavior** — draft vs final, critic delta. Standard harnesses do **not** run your 3-stage pipeline; use `eval_triad_delta.py` or a future full-triad task script.

---

## Core benchmark matrix (GPU)

Default: **five** separate `lm_eval run` invocations (GSM8K is **0-shot only**; 8-shot was dropped to reduce prompt interference).

| Run | Task / group | Shots |
|-----|----------------|------|
| GSM8K (0-shot) | `gsm8k` | 0 |
| HellaSwag (5-shot) | `hellaswag` | 5 |
| ARC-Challenge (10-shot) | `arc_challenge` | 10 |
| MMLU mini (5-shot, 8 subjects) | `mmlu_mini_subset` | 5 |
| BigBench Hard (0-shot, CoT) | `bbh_cot_zeroshot` | 0 |

Subjects for the MMLU mini row (see `configs/mmlu_mini_subset.yaml`): high school math, elementary math, conceptual physics, college CS, high school CS, miscellaneous, global facts, world history. Results JSON includes **per-subject** scores and a **group** aggregate. Edit that YAML to swap subjects (task ids must match `lm_eval`, e.g. `mmlu_*`). For **full** 57-subject MMLU, use `--tasks mmlu` in a separate run (much slower).

```bash
pip install -r requirements-eval.txt

python scripts/benchmark_suite.py outputs/merged_reasoning
# or: bash scripts/run_benchmark_suite.sh outputs/merged_reasoning
```

Results land under **`eval/benchmark_suite_<UTC-timestamp>/`**: one JSON per run (`gsm8k_0shot.json`, …) plus `manifest.json`. Use `--output-dir` to choose a fixed folder.

Smoke test (not publishable): `--limit 10`

Copy the whole directory off the GPU, then fill `data/benchmark_results_template.json`.

**Pack repo + adapters for upload:**

```bash
bash scripts/pack_for_gpu_eval.sh
```

On Linux, if `tar` fails with **cannot change ownership** (archives created on macOS), extract with:

`tar xzf miso-mini-gpu-eval-*.tar.gz --no-same-owner`

Then `pip install -r requirements-eval.txt`, merge adapters, run `benchmark_suite.py`, download the `eval/` folder, delete `outputs/merged_*` and HF cache if you need disk.

---

## Legacy: one-shot 0-shot trio (gsm8k + mmlu + arc_challenge)

Older workflow: a **single** lm-eval run with `--num_fewshot 0` for all tasks.

```bash
python scripts/benchmark_suite.py --legacy outputs/merged_reasoning
# or: bash scripts/run_benchmark_suite_0shot.sh outputs/merged_reasoning
```

---

## Three “modes” (what each number means)

| Mode | What gets scored | How |
|------|------------------|-----|
| **Response-only** | Single LoRA: chat-style answers | Merge `triad_adapters/response` → `benchmark_suite.py outputs/merged_response` |
| **Reasoning-only** | Single LoRA: reasoning head | Merge `triad_adapters/reasoning` → `outputs/merged_reasoning` + same script |
| **Full triad** | Reasoning → response → critic | **Not** one merged HF model; see `scripts/eval_triad_task.py` (stub) |

```bash
bash scripts/merge_eval_modes.sh   # builds merged_response + merged_reasoning
```

---

## Option A — Merge + lm-eval

```bash
pip install -r requirements-eval.txt

python scripts/merge_adapter_for_eval.py \
  --adapter outputs/triad_adapters/reasoning \
  --output outputs/merged_reasoning
```

**Shot counts:** the default `benchmark_suite.py` matrix sets `--num_fewshot` per run (see table above). If you call `lm_eval` by hand, match those flags to your table.

---

## Option B — Triad delta (draft vs final)

```bash
python scripts/convert_peft_to_mlx.py   # if not already done

python scripts/eval_triad_delta.py \
  --questions data/eval_questions_sample.jsonl \
  --adapters-dir outputs/triad_adapters_mlx \
  --out eval/triad_runs.jsonl
```

---

## Option C — Full triad on GSM8K (advanced)

lm-eval loads **one** model per run. For triad-wide task scores, use a custom loop or `eval_triad_task.py` when implemented.

---

## Plot: triad critic (optional)

After `eval_triad_delta.py` produces JSONL:

```bash
python scripts/plot_triad_delta.py \
  --input eval/triad_runs.jsonl \
  --output outputs/figures/triad_critic_delta.png
```

Optional scatter plots: `scripts/plot_benchmark_scatter.py` — supply your own JSON with measured points only; the repo does not ship leaderboard scatter figures.

---

## Practical order

1. `pack_for_gpu_eval.sh` → GPU → merge → `benchmark_suite.py` → download `eval/benchmark_suite_*`.
2. `eval_triad_delta.py` for critic effect on your JSONL.
