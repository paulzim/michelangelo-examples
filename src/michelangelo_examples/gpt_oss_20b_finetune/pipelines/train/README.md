# gpt_oss_20b_finetune: Train

LoRA fine-tunes a GPT causal-LM checkpoint on an instruction-following
dataset (Alpaca by default), running against a released
[`michelangelo`](https://pypi.org/project/michelangelo/) PyPI package -- no
core `michelangelo` monorepo checkout required.

Part of the `gpt_oss_20b_finetune` project (use case); this is its only
pipeline.

### Prerequisites

- **Python 3.11+.** Both entry points below (`__main__.py` and
  `pipeline.py`) raise a `RuntimeError` at import time on older
  interpreters -- see the project [README](../../README.md#requires-python-311)
  for why this project's floor differs from this repo's other projects.

## Quick start (local)

```bash
pip install "michelangelo-examples[gpt-oss-20b-finetune]"
python -m michelangelo_examples.gpt_oss_20b_finetune.pipelines.train
```

This trains GPT-2 with LoRA on a small (~32-sample) slice of the Stanford
Alpaca dataset in a couple of minutes on a laptop CPU -- no Ray, no MLflow,
no sandbox required. GPT-2 is used as a CPU-feasible proxy; the same model
code (`model.py`) supports any HuggingFace causal-LM checkpoint, including
GPT-OSS-20B on a GPU-backed cluster via the full pipeline below.

## Full pipeline

```
prepare_finetune_dataset  ->  simple_train_gpt  ->  evaluate_gpt_model
        (Ray)                     (Ray)                  (Ray)
```

| Step | File | Runtime | Description |
|---|---|---|---|
| `prepare_finetune_dataset` | `data.py` | Ray | Load, format, and tokenize an instruction-following dataset (Alpaca/Dolly/oasst1) |
| `simple_train_gpt` | `train.py` | Ray | LoRA fine-tune via Ray Train + PyTorch Lightning, logging to MLflow |
| `evaluate_gpt_model` | `eval.py` | Ray | Perplexity and generation-quality metrics on the held-out test split |

### End-to-end: sandbox to running pipeline

```bash
# 1. Create a sandbox (skip if you already have one running)
ma sandbox create

# 2. Register the gpt-oss-20b-finetune project (namespace: gpt-oss-20b-finetune)
ma project apply -f src/michelangelo_examples/gpt_oss_20b_finetune/config/project.yaml

# 3. Register this pipeline (namespace: gpt-oss-20b-finetune, name: gpt-oss-20b-finetune-train)
ma pipeline apply -f src/michelangelo_examples/gpt_oss_20b_finetune/pipelines/train/pipeline.yaml

# 4. Run it
ma pipeline run -n gpt-oss-20b-finetune --name gpt-oss-20b-finetune-train
```

`ma pipeline run` dispatches through Cadence using the image already
declared in `pipeline.yaml`'s `michelangelo/uniflow-image` annotation
(`ghcr.io/michelangelo-ai/michelangelo-examples:gpt-oss-20b-finetune`,
built by this repo's own CI) -- no `--image`/`--environ` flags needed for
this path. Use `remote-run` (below) instead of `ma pipeline apply` +
`ma pipeline run` if you need to override the image or pass environment
variables without registering the pipeline first.

## How It Works

### LoRA, not full fine-tuning

`model.py`'s `GPTLightningModule` wraps the base causal-LM with
[PEFT](https://github.com/huggingface/peft)'s `LoraConfig`
(`target_modules=["c_attn", "c_proj"]`, GPT-2's attention/projection
layers) when `use_lora=True` (the default in both entry points) --
typically well under 2% of parameters are trainable, making this feasible
on modest hardware even for a large base model.

### `simple_train_gpt`'s MLflow logging

Unlike this repo's other pipelines, `train.py` logs training metrics via
`pytorch_lightning.loggers.MLFlowLogger` against `MLFLOW_TRACKING_URI`
(defaults to `http://localhost:5001` if unset) -- ported as-is from core's
`simple_train.py`. This only affects the full `pipeline.py` path; the
local `__main__.py` runner does not use MLflow at all.

### Checkpoint hand-off between `train` and `eval`

`simple_train_gpt` returns an S3 URI for the Ray/Lightning checkpoint
(`{checkpoint_path}/model_checkpoint.ckpt`); `evaluate_gpt_model` reads it
back via `fsspec` and reconstructs the model with `create_gpt_model()`
before loading the checkpoint's `state_dict`.

## Remote Run

Pass environment variables via `--environ` flags -- they are serialized into
the Cadence/Temporal workflow and injected into every task's runtime
environment, reaching remote workers. Shell `export` statements before the
command only affect the local launcher and do not propagate.

```bash
python -m michelangelo_examples.gpt_oss_20b_finetune.pipelines.train.pipeline \
  remote-run \
  --image ghcr.io/michelangelo-ai/michelangelo-examples:gpt-oss-20b-finetune \
  --storage-url s3://your-bucket/workflows \
  --environ AWS_ENDPOINT_URL=http://your-minio-endpoint:9000 \
  --environ AWS_ACCESS_KEY_ID=your-access-key \
  --environ AWS_SECRET_ACCESS_KEY=your-secret-key \
  --environ MLFLOW_TRACKING_URI=http://your-mlflow-endpoint:5001 \
  --yes
```

### k3d sandbox

```bash
python -m michelangelo_examples.gpt_oss_20b_finetune.pipelines.train.pipeline \
  remote-run \
  --image ghcr.io/michelangelo-ai/michelangelo-examples:gpt-oss-20b-finetune \
  --storage-url s3://michelangelo/workflows \
  --environ AWS_ENDPOINT_URL=http://minio:9091 \
  --environ AWS_ACCESS_KEY_ID=michelangeloadmin \
  --environ AWS_SECRET_ACCESS_KEY=michelangeloadmin \
  --environ MLFLOW_TRACKING_URI=http://mlflow-proxy:5001 \
  --yes
```

The `ghcr.io/michelangelo-ai/michelangelo-examples:gpt-oss-20b-finetune`
image is built by this repo's own CI
(`.github/workflows/build-image.yaml`) from the root `Dockerfile` -- no
need to build it yourself unless testing a local change:

```bash
docker build --build-arg PROJECT_EXTRA=gpt-oss-20b-finetune -t michelangelo-examples:gpt-oss-20b-finetune-local .
k3d image import michelangelo-examples:gpt-oss-20b-finetune-local -c michelangelo-sandbox
kubectl delete cachedoutputs --all   # clear stale cached task outputs
```

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | No | `http://localhost:5001` | MLflow tracking server used by `train.py`'s `MLFlowLogger` |
| `AWS_ENDPOINT_URL` | No | -- | S3-compatible endpoint for Ray checkpoint storage (`s3://default/ray_checkpoints`) |
| `AWS_ACCESS_KEY_ID` | If `AWS_ENDPOINT_URL` set | -- | Access key ID |
| `AWS_SECRET_ACCESS_KEY` | If `AWS_ENDPOINT_URL` set | -- | Secret access key |

## Network egress note

`prepare_finetune_dataset`/`load_alpaca_dataset` (and the local
`__main__.py` runner) download the Stanford Alpaca dataset from the
HuggingFace Hub at run time, not just install time -- if this fails (e.g. a
network-restricted sandbox), all three dataset loaders fall back to a
small offline dummy instruction dataset (see `data.py`'s
`create_dummy_dataset`) so the pipeline still completes end to end.
