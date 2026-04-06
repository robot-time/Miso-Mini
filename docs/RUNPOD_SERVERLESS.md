# RunPod Serverless — triad (3× LoRA)

The triad is **three PEFT adapters** on one base (`microsoft/Phi-4-mini-instruct` by default). Serverless workers should load the **base once** and **swap adapters** (`TriadHotSwapRuntime` in `scripts/infer_triad.py`), not reload the full model three times per request.

## What you ship

| Asset | Role |
|--------|------|
| `reasoning/`, `response/`, `critic/` | LoRA folders (same layout as `outputs/triad_adapters/`) |
| `configs/triad_reasoning.yaml` | `model_name` + `load_in_4bit` (must match how you trained) |
| Docker image | **`Dockerfile` at repo root** + `serverless/handler.py` |

## Build the image

From the **repo root**:

```bash
docker build -t YOUR_REGISTRY/miso-triad-serverless:latest .
```

RunPod **GitHub** integration: set the Dockerfile path to **`Dockerfile`** (repository root) and build context to the **repo root** (`.`). Do **not** use `serverless/` as the context directory, or `COPY serverless/...` will fail.

**Adapters:** either:

1. **Network Volume (recommended)** — upload `outputs/triad_adapters` to a RunPod volume and mount it at e.g. `/runpod/triad`; set **`TRIAD_ADAPTERS_DIR=/runpod/triad`**, or  
2. **Bake in** — uncomment the `COPY outputs/triad_adapters/...` lines in the root `Dockerfile` and rebuild (image gets large).

## RunPod console

1. Create a **Serverless** endpoint; use your pushed image from Docker Hub / GHCR.
2. GPU: **≥24 GB** is comfortable for Phi-4-mini + 4-bit base + three LoRAs in one process (adjust if you use full precision).
3. Set environment variables if needed:
   - `TRIAD_ADAPTERS_DIR` — mount path to the folder that contains `reasoning/`, `response/`, `critic/`
   - `TRIAD_CONFIG` — optional override path to YAML inside the image
   - `HF_TOKEN` — if the base model is gated
   - `HF_HOME` — point to a **large** path (see disk section below) or [model caching](https://docs.runpod.io/serverless/endpoints/model-caching)

### Disk space (`No space left on device` / `File reconstruction error`)

The base model download + Hugging Face cache can need **well over 10 GB** on disk. Serverless **container disk** is easy to exhaust.

Do one or more of:

1. **Raise container disk** for the endpoint (RunPod **Edit endpoint** → advanced / storage — set **at least ~30 GB** if the UI offers it).
2. Attach a **Network volume** (e.g. 30–50 GB), mount it on the worker, then set:
   - `HF_HOME=/path/to/volume/hf_home` (use the mount path RunPod shows for that volume)
   - Optionally `TMPDIR=/path/to/volume/tmp` so temp extraction also uses the volume.
3. Use RunPod **[cached models](https://docs.runpod.io/serverless/endpoints/model-caching)** for `microsoft/Phi-4-mini-instruct` when available so workers skip a full download.

The image sets **`HF_HUB_DISABLE_XET=1`** so Hub downloads use a simpler path that is less likely to blow temp space during XET reconstruction (still need enough disk overall).

## Request / response shape

**Input** (`POST` body `input` object):

```json
{
  "question": "Why is the sky blue?",
  "max_new_tokens": 384
}
```

Also accepts `prompt` or `message` instead of `question`.

**Output:**

```json
{
  "user": "...",
  "reasoning": "...",
  "draft": "...",
  "final": "..."
}
```

On error: `{ "error": "...", "type": "..." }`.

## Local smoke test (GPU machine)

Mount adapters from this repo (image still starts the RunPod worker entrypoint):

```bash
docker run --gpus all \
  -e TRIAD_ADAPTERS_DIR=/adapters \
  -v "$(pwd)/outputs/triad_adapters:/adapters:ro" \
  YOUR_REGISTRY/miso-triad-serverless:latest
```

(Image must be built from repo root with the root `Dockerfile`.)

Use RunPod’s [local serverless testing](https://docs.runpod.io/serverless/development/local-testing) or hit the deployed endpoint from the console.

## CLI parity (no Docker)

```bash
source .venv/bin/activate
python scripts/infer_triad.py --hot-swap --json --quiet \
  --adapters-dir outputs/triad_adapters \
  --question "Hello"
```

## References

- [RunPod — create Dockerfile](https://docs.runpod.io/serverless/workers/create-dockerfile)  
- [RunPod — deploy worker](https://docs.runpod.io/serverless/workers/deploy)
