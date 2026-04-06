# RunPod (cloud GPU) — upload, install, train

RunPod is a normal Linux host with an NVIDIA GPU. This repo does not need a special “RunPod format”: use the same scripts as locally, with CUDA so **Unsloth** and **4-bit QLoRA** work.

## 1. Create a pod

- **Template:** PyTorch + CUDA (match the image’s CUDA to what `torch` expects; default PyTorch images are usually fine).
- **GPU:** RTX 4090 or 3090 is a good cost/performance balance.
- **Disk:** 20–40 GB is enough for the base model cache + adapters (you are not pre-training from scratch).

## 2. Get the code onto the pod

Pick one.

### A. Git (recommended if the project is on GitHub)

```bash
git clone <YOUR_REPO_URL> Miso-Mini
cd Miso-Mini
```

### B. Tarball from your Mac (this repo)

On your **Mac**, from the project directory:

```bash
bash scripts/pack_for_runpod.sh
```

This writes `miso-mini-runpod-YYYYMMDD-HHMM.tar.gz` (excludes `.venv`, `outputs/`, `openai_key.txt`, `.git`).

**Eval-only (LoRAs + benchmark scripts, smaller upload):** from the repo root run `bash scripts/pack_for_gpu_eval.sh` — produces `miso-mini-gpu-eval-*.tar.gz` including `outputs/triad_adapters/`. See `docs/BENCHMARK.md`.

Upload the file (Jupyter drag-and-drop, or `scp`), then on the pod:

```bash
mkdir -p ~/work && cd ~/work
tar xzvf miso-mini-runpod-*.tar.gz
cd Miso-Mini   # if the archive unpacked to this folder name; adjust if needed
```

### C. `scp` the whole folder (no tarball)

From your Mac (replace host and path):

```bash
scp -r /path/to/Miso-Mini root@<POD_IP>:/root/work/
```

## 3. Install dependencies

On the pod, from the repo root:

```bash
bash scripts/runpod_install.sh
source .venv/bin/activate
```

Confirm CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

You should see `True` and a GPU name.

### Updating code on a pod you already set up

You **do not** reinstall PyTorch or recreate `.venv` unless `requirements.txt` changed.

**If you use Git on the pod:**

```bash
cd /workspace/Miso-Mini   # or wherever you cloned (RunPod often uses /workspace)
git pull
# Only if requirements changed:
source .venv/bin/activate
pip install -r requirements.txt
```

**If you use upload (no Git):** on your Mac, run `bash scripts/pack_for_runpod.sh` again, upload the new `.tar.gz` to the pod (e.g. under `/workspace/`).

**macOS tar warnings:** `LIBARCHIVE.xattr.com.apple.provenance` is harmless. Newer packs set `COPYFILE_DISABLE=1` on the Mac to reduce this. Ignore the lines if you still see them.

**GPU eval tarball (`pack_for_gpu_eval.sh`):** if Linux `tar` exits non-zero with **Cannot change ownership to uid …** (metadata from a Mac-created archive), extract with `tar xzf FILE.tar.gz --no-same-owner` (or add that flag to your unpack script so `set -e` pipelines do not abort).

**Option A0 — Copy merge (no rsync, no helper script):** set `ARCH` to your real file (`ls /workspace/*.tar.gz`).

```bash
ARCH=/workspace/miso-mini-runpod-YYYYMMDD-HHMM.tar.gz
TMP=$(mktemp -d)
tar xzf "$ARCH" -C "$TMP"
while IFS= read -r -d '' item; do
  name=$(basename "$item")
  [[ "$name" == ".venv" ]] && continue
  [[ "$name" == "outputs" ]] && continue
  rm -rf "/workspace/Miso-Mini/$name"
  cp -a "$item" "/workspace/Miso-Mini/"
done < <(find "$TMP/Miso-Mini" -mindepth 1 -maxdepth 1 -print0)
rm -rf "$TMP"
```

**Option A — Apply bundle with helper script** (included in newer packs): many pods do **not** have `rsync`. After `scripts/apply_runpod_bundle.sh` exists under `/workspace/Miso-Mini/scripts/`:

```bash
ls /workspace/*.tar.gz
bash /workspace/Miso-Mini/scripts/apply_runpod_bundle.sh /workspace/miso-mini-runpod-YYYYMMDD-HHMM.tar.gz
cd /workspace/Miso-Mini && source .venv/bin/activate
pip install -r requirements.txt   # only if requirements.txt changed
```

If `apply_runpod_bundle.sh` is missing on the pod, install rsync and use **Option A2**, or **Option B**.

**Option A2 — `rsync`:** `apt-get update && apt-get install -y rsync` (Debian/Ubuntu-based pod), then:

```bash
tar xzf /workspace/miso-mini-runpod-NEW.tar.gz -C /tmp
rsync -a --exclude='.venv' --exclude='outputs' /tmp/Miso-Mini/ /workspace/Miso-Mini/
rm -rf /tmp/Miso-Mini
```

Use the **real filename** (tab-complete or `ls`). Do **not** run `tar` from `/` unless the `.tar.gz` is in `/` — it is usually in `/workspace/`.

**Option B — Full replace:** `rm -rf Miso-Mini`, unpack tarball under `/workspace`, then `bash scripts/runpod_install.sh` again (your Hugging Face **model cache** under `~/.cache/huggingface` is unchanged, so the base model usually does not re-download; pip may still reinstall packages from cache quickly).

**Rule of thumb:** do **not** blindly delete `outputs/` if you need saved adapters. Re-run `pip install -r requirements.txt` only when `requirements.txt` changes.

## 4. Hugging Face access

If downloads for `microsoft/Phi-4-mini-instruct` fail (gated or rate limits):

```bash
pip install huggingface_hub
huggingface-cli login
```

Paste a token with **read** access from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## 5. Data on the pod

Triad training expects:

- `data/miso_raw.jsonl` (for the critic path via `convert_to_sft.py`)
- WildChat-derived `data/response_sft.jsonl` unless you already built it
- **Reasoning:** run `python scripts/gsm8k_to_reasoning_sft.py --output data/reasoning_gsm8k.jsonl` *before* `setup_triad_data.sh` if you want full GSM8K (~7.5k rows) merged with the tiny seed; otherwise you only get the seed examples.

**If you packed the repo with `pack_for_runpod.sh`**, copy your `data/` (including `miso_raw.jsonl`) into the project before packing, or upload a separate `data` tarball and merge into `Miso-Mini/data/`.

Then:

```bash
source .venv/bin/activate
export WILDCHAT_MAX_ROWS=8000   # optional; trim WildChat size for a quicker first run
./scripts/setup_triad_data.sh
```

If WildChat download is slow or blocked, prepare `data/reasoning_sft.jsonl`, `data/response_sft.jsonl`, and `data/critic_sft.jsonl` locally and upload only those files into `data/`.

## 6. Train the three LoRAs (Unsloth on CUDA)

Force Unsloth when CUDA is present (the default `train_triad_adapters.sh` already picks Unsloth if `torch.cuda.is_available()`):

```bash
source .venv/bin/activate
export FORCE_TRAINER=unsloth
./scripts/train_triad_adapters.sh
```

**Auto-halt after success (recommended on RunPod):** training only charges GPU time while the pod runs. This repo does **not** shut down the machine by default. Use the wrapper so the host halts **only if** setup and training both exit 0 (same idea as `./scripts/train_triad_adapters.sh && shutdown -h now`, but in one place):

```bash
source .venv/bin/activate
export FORCE_TRAINER=unsloth
./scripts/run_triad_on_runpod.sh
```

- `SKIP_SHUTDOWN=1` — run training but do not halt (testing).
- `RUN_SETUP=0` — skip `./scripts/setup_triad_data.sh` if `data/*_sft.jsonl` are already present.
- `HALT_DELAY_SEC=120` — optional sleep before `shutdown -h now` so you can start a download.

If `shutdown` is missing in the container, stop the pod from the RunPod UI.

Outputs:

- `outputs/triad_adapters/reasoning/`
- `outputs/triad_adapters/response/`
- `outputs/triad_adapters/critic/`

Smoke-test inference:

```bash
python scripts/infer_triad.py --adapters-dir outputs/triad_adapters --question "Hello" --json --quiet
```

**RunPod Serverless (HTTP triad endpoint):** see [RUNPOD_SERVERLESS.md](RUNPOD_SERVERLESS.md) — root `Dockerfile`, `serverless/handler.py`, hot-swap single base + 3 LoRAs.

### Unsloth slow at startup or you hit Ctrl+C by mistake

The first `FastLanguageModel.from_pretrained` call can sit for a while while Unsloth patches transformers (**do not** assume it is frozen; wait a few minutes). If you see `KeyboardInterrupt` in `unsloth_compile_transformers` / `patch_gpt_oss`, that is usually **you** stopping the process, not a crash.

**If Unsloth errors or never finishes loading:** use the Hugging Face trainer instead (same YAML and data, typically a bit slower, fewer Unsloth-specific patches):

```bash
export FORCE_TRAINER=hf
./scripts/train_triad_adapters.sh
```

**HF Hub rate limit:** set `HF_TOKEN` or run `huggingface-cli login` (you already saw the unauthenticated warning during WildChat download).

**Torch / cpp extensions warning:** Unsloth may warn about `torch` version vs bundled C++ extensions; upgrading `torch` in the venv can clear it, or use `FORCE_TRAINER=hf` if training otherwise works.

**Tokenizing dataset: `cannot pickle 'ConfigModuleInstance'` / `NoneType` during `dataset.map`:** TRL + `datasets` may use many worker processes (`num_proc=64`), which pickles the tokenizer and fails with Unsloth/HF. The training scripts default to **`dataset_num_proc=1`** (set `DATASET_NUM_PROC` in the environment to raise it only if you know pickling works).

## 7. Download adapters back to your Mac

From your Mac:

```bash
scp -r root@<POD_IP>:/root/work/Miso-Mini/outputs/triad_adapters ./triad_adapters_backup/
```

Or zip on the pod and use Jupyter’s file browser to download.

## 8. Stop or delete the pod

Stop the pod when training finishes so you are not billed for idle GPU time. Prefer `./scripts/run_triad_on_runpod.sh` so a successful run can halt the host automatically; still set a calendar reminder to check the dashboard in case the job failed early or `shutdown` is unavailable.

---

**Single adapter / curriculum** (non-triad): use `./scripts/run_best.sh` or the scripts in `README.md` the same way after install; still prefer `FORCE_TRAINER=unsloth` on CUDA.
