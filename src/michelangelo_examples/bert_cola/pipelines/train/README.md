# bert_cola: Train

Fine-tunes `bert-base-cased` for linguistic acceptability classification on
the Corpus of Linguistic Acceptability (CoLA) task from the GLUE benchmark,
running against a released [`michelangelo`](https://pypi.org/project/michelangelo/)
PyPI package -- no core `michelangelo` monorepo checkout required.

Part of the `bert_cola` project (use case); this is its only pipeline.

## Quick start

```bash
pip install "michelangelo-examples[bert-cola]"
python -m michelangelo_examples.bert_cola.pipelines.train.pipeline
```

This runs `train_workflow` locally via `ctx.run(...)` -- real BERT
fine-tuning, packaging, and push, using local temp-directory storage and an
in-memory registry (no sandbox required, but still real Ray execution, not
a Ray-free plain-Python shortcut).

## Full pipeline

```
load_data  →  train  →  assembler  →  push_step
   (Ray)       (Ray)       (Ray)        (Ray)
```

| Step | File | Runtime | Description |
|---|---|---|---|
| `load_data` | `data.py` | Ray | Load + tokenize the CoLA dataset from GLUE |
| `train` | `train.py` | Ray | Fine-tune BERT via HuggingFace `Trainer`, upload as a `ModelVariable` |
| `assembler` | `assembler.py` | Ray | Package the fine-tuned checkpoint into deployable + raw Triton packages via `custom_assembler` |
| `push_step` | `push.py` | Ray | Push the assembled model to storage/registry |

### Prerequisites

- A Michelangelo sandbox running (`ma sandbox create`)
- The `bert-cola` project applied from this project's own config:
  `ma project apply -f src/michelangelo_examples/bert_cola/config/project.yaml`
- Python 3.10+

### End-to-end: sandbox to running pipeline

```bash
# 1. Create a sandbox (skip if you already have one running)
ma sandbox create

# 2. Register the bert-cola project (namespace: bert-cola)
ma project apply -f src/michelangelo_examples/bert_cola/config/project.yaml

# 3. Register this pipeline (namespace: bert-cola, name: bert-cola-train)
ma pipeline apply -f src/michelangelo_examples/bert_cola/pipelines/train/pipeline.yaml

# 4. One-time: start the daily-noon-run trigger declared in pipeline.yaml
#    (registering the pipeline in step 3 does NOT start it on its own --
#    this TriggerRun must be explicitly created once)
ma trigger_run create --namespace=bert-cola --pipeline=bert-cola-train --trigger-name=daily-noon-run

# 5. Run it manually (optional -- the trigger above already runs it daily at 12:00pm)
ma pipeline run -n bert-cola --name bert-cola-train
```

Step 4 above is a **one-time** command per trigger -- it registers the
`daily-noon-run` trigger (declared in `pipeline.yaml`'s `triggerMap`) with
the platform so it starts producing runs on its own cron schedule
(`0 12 * * *`, i.e. daily at 12:00pm). It is distinct from step 5's
`ma pipeline run`, which triggers one ad hoc manual run and can be repeated
any number of times independently of the daily schedule.

`ma pipeline run` dispatches through Cadence using the image already
declared in `pipeline.yaml`'s `michelangelo/uniflow-image` annotation
(`ghcr.io/michelangelo-ai/michelangelo-examples:bert-cola`, built by this
repo's own CI) -- no `--image`/`--environ` flags needed for this path. Use
`remote-run` (below) instead of `ma pipeline apply` + `ma pipeline run` if
you need to override the image or pass environment variables without
registering the pipeline first.

## How It Works

### Custom (Python-backend) assembler, not `torch_assembler`

BERT's multi-input `forward()` (`input_ids`/`attention_mask`/
`token_type_ids`) isn't a good TorchScript-tracing fit, so `model.py`
defines `BertColaModel`, a `michelangelo.lib.model_manager.interface.custom_model.Model`
subclass, and `assembler.py` uses `custom_assembler` instead of
`torch_assembler`. `train()` wraps the fine-tuned model as a
`ModelVariable` (auto-detects the custom training framework since
`BertColaModel` implements the `CustomModel` interface) instead of
returning a local path -- `train()` and `assembler()` aren't guaranteed to
run on the same machine. `assembler()` downloads it via the same fsspec
mechanism `ModelVariable` itself uses, then re-uploads it through its own
storage backend before handing it to `custom_assembler()`.

### `tar_deployable_package`

`push.py` sets `ModelPluginConfig(tar_deployable_package=True)` so the
deployable Triton package is pushed as a single `model.tar` rather than
loose files.

## Remote Run

Pass environment variables via `--environ` flags -- they are serialized into the
Cadence/Temporal workflow and injected into every task's runtime environment,
reaching remote workers. Shell `export` statements before the command only
affect the local launcher and do not propagate.

```bash
python -m michelangelo_examples.bert_cola.pipelines.train.pipeline \
  remote-run \
  --image ghcr.io/michelangelo-ai/michelangelo-examples:bert-cola \
  --storage-url s3://your-bucket/workflows \
  --environ AWS_ENDPOINT_URL=http://your-minio-endpoint:9000 \
  --environ AWS_ACCESS_KEY_ID=your-access-key \
  --environ AWS_SECRET_ACCESS_KEY=your-secret-key \
  --environ REGISTRY_ENDPOINT=your-apiserver-host:15566 \
  --yes
```

### k3d sandbox

```bash
python -m michelangelo_examples.bert_cola.pipelines.train.pipeline \
  remote-run \
  --image ghcr.io/michelangelo-ai/michelangelo-examples:bert-cola \
  --storage-url s3://michelangelo/workflows \
  --environ AWS_ENDPOINT_URL=http://minio:9091 \
  --environ AWS_ACCESS_KEY_ID=michelangeloadmin \
  --environ AWS_SECRET_ACCESS_KEY=michelangeloadmin \
  --environ REGISTRY_ENDPOINT=michelangelo-apiserver:15566 \
  --yes
```

The `ghcr.io/michelangelo-ai/michelangelo-examples:bert-cola` image is
built by this repo's own CI (`.github/workflows/build-image.yaml`) from the
root `Dockerfile` -- no need to build it yourself unless testing a local
change:

```bash
docker build --build-arg PROJECT_EXTRA=bert-cola -t michelangelo-examples:bert-cola-local .
k3d image import michelangelo-examples:bert-cola-local -c michelangelo-sandbox
kubectl delete cachedoutputs --all   # clear stale cached task outputs
```

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `AWS_ENDPOINT_URL` | No | -- | S3-compatible endpoint URL (include scheme, e.g. `http://minio:9091`). Unset -> local storage |
| `AWS_ACCESS_KEY_ID` | If `AWS_ENDPOINT_URL` set | -- | Access key ID |
| `AWS_SECRET_ACCESS_KEY` | If `AWS_ENDPOINT_URL` set | -- | Secret access key |
| `AWS_S3_BUCKET` | No | Parsed from `MA_FILE_SYSTEM` or `UF_STORAGE_URL` | Target bucket name |
| `REGISTRY_ENDPOINT` | No | -- | Model registry gRPC endpoint (`host:port`). Unset -> in-memory only |
| `REGISTRY_INSECURE` | No | `true` | Set `false` to enable TLS for the registry connection |
| `REGISTRY_NAMESPACE` | No | `MA_NAMESPACE` (the pipeline's own namespace), else `default` | Model registry namespace |

## Deploying to Triton

See [`../../inferenceserver/`](../../inferenceserver/) for this project's
deploy manifests. In short:

```bash
ma inference_server apply -f src/michelangelo_examples/bert_cola/inferenceserver/inferenceserver.yaml
# then, with desiredRevision.name set to the model name printed by push_step:
ma deployment apply -f src/michelangelo_examples/bert_cola/inferenceserver/deployment.yaml
```
