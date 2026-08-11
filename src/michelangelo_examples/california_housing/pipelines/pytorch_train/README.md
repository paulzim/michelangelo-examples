# California Housing: PyTorch Lightning Train

End-to-end ML pipeline for California Housing price prediction using PyTorch
Lightning via `tabular_trainer`'s `train_tabular()`, running against a
released [`michelangelo`](https://pypi.org/project/michelangelo/) PyPI
package — no core `michelangelo` monorepo checkout required.

Part of the `california_housing` project (use case); this is its
`pytorch_train` pipeline. For the XGBoost variant, see
[`xgb_train/`](../xgb_train/README.md).

Two ways to run this pipeline:

- **Local runner** (`python -m ...`, see Quick start) — trains
  `TorchRegressionModel` locally on a data slice in a couple of minutes, no
  Ray/Spark/Cadence/sandbox required. Start here.
- **The full pipeline** (`pipeline.py` + `pipeline.yaml`) —
  the production-shaped, Cadence-dispatched version: feature prep → Spark
  preprocessing → distributed Ray Train → push to storage/registry.
  Requires a running Michelangelo sandbox.

## Quick start (local)

```bash
pip install "michelangelo-examples[california-housing]"
python -m michelangelo_examples.california_housing.pipelines.pytorch_train
```

## Full pipeline

```
feature_prep  →  preprocess  →  train  →  push_step
   (Ray)           (Spark)      (Ray)      (Spark)
```

| Step | File | Runtime | Description |
|---|---|---|---|
| `feature_prep` | `feature_prep.py` | Ray | Load dataset, train/test split, Ray Datasets |
| `preprocess` | `preprocess.py` | Spark | Cast columns to float |
| `train` | `train.py` | Ray | Distributed Lightning training via `tabular_trainer.train_tabular()` |
| `push_step` | `push.py` | Spark | Push model and preprocessed datasets to storage/registry |

### Prerequisites

- A Michelangelo sandbox running (`ma sandbox create`)
- The `california-housing` project applied from this project's own config:
  `ma project apply -f src/michelangelo_examples/california_housing/config/project.yaml`
- Python 3.10+
- Java 17 with `JAVA_HOME` set — required for Spark. Java 21 is incompatible
  with PySpark 3.5 + Hadoop 3.3 (`getSubject is not supported`). On macOS:
  `brew install openjdk@17` then
  `export JAVA_HOME=$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home`

If you hit sandbox issues getting to this point (helm timeouts, zombie Ray
clusters, Cadence domain registration, MA Studio 415 errors, ...), see
[`docs/sandbox-troubleshooting.md`](../../../../../docs/sandbox-troubleshooting.md).

### End-to-end: sandbox to running pipeline

The full command sequence a first-time operator needs, in order — each
step depends on the one before it:

```bash
# 1. Create a sandbox (skip if you already have one running)
ma sandbox create

# 2. Register the california-housing project (namespace: california-housing)
ma project apply -f src/michelangelo_examples/california_housing/config/project.yaml

# 3. Register this pipeline (namespace: california-housing, name: pytorch-train)
ma pipeline apply -f src/michelangelo_examples/california_housing/pipelines/pytorch_train/pipeline.yaml

# 4. Run it
ma pipeline run -n california-housing --name pytorch-train
```

`ma pipeline run` dispatches through Cadence using the image already
declared in `pipeline.yaml`'s `michelangelo/uniflow-image` annotation
(`ghcr.io/michelangelo-ai/michelangelo-examples:california-housing`, built
by this repo's own CI) — no `--image`/`--environ` flags needed for this
path. Use `remote-run` (below) instead of `ma pipeline apply` +
`ma pipeline run` if you need to override the image or pass environment
variables without registering the pipeline first.

## How It Works

### `train_tabular()` instead of a bespoke training loop

This example's `train.py` is a thin `@uniflow.task` wrapper around
`michelangelo.workflow.tasks.tabular_trainer.task.train_tabular()` — the
shared Lightning + Ray Train dispatcher. `train_tabular()` builds its own
multi-node-safe `RunConfig` internally and returns a `ModelVariable`
pointing at the uploaded checkpoint, rather than a raw checkpoint path or an
assembled `ModelArtifact`. `push.py` downloads that checkpoint locally (via
`fsspec.core.url_to_fs()`) and wraps it in a `ModelArtifact` before handing
it to `ModelPusherPlugin`, since no OSS "packager" task exists yet to do
that conversion automatically.

### `TorchRegressionModel` — the model to plug in

`train_tabular()` requires a `LightningModule` subclass, referenced by dotted
import path via `LightningTrainerConfig.model_class`. This example defines a
minimal MLP regressor in [`model.py`](model.py): two hidden `nn.Linear`
layers (64 → 32, ReLU) feeding a single scalar output, trained with
`MSELoss` and `Adam`. The local runner (`__main__.py`) reuses the exact
same model class.

Ray Data batches passed to `training_step`/`validation_step` (and the
local runner's plain `DataLoader` batches) are dicts of column-name →
tensor. `TorchRegressionModel` is constructed with
`feature_columns`/`label_column` (via `LightningTrainerConfig.model_kwargs`
in the full pipeline, or directly in the local runner) so it knows which
batch keys to stack into its input tensor and which key holds the
regression target.

### CPU-only precision

`train.py` explicitly forces `precision="32"` via `LightningTrainerKwargs`
rather than relying on `train_tabular()`'s dispatcher default
(`"bf16-mixed"`). Verified locally that `bf16-mixed` does not error on a
CPU-only accelerator — Lightning runs real `torch.autocast('cpu',
dtype=torch.bfloat16)` AMP, not a silent fallback — but on x86 CPUs without
`AVX512_BF16` this runs via slower software emulation. `precision="32"`
keeps this tutorial-oriented example's runs fast and deterministic on the
k3d sandbox's CPU-only nodes.

### No eval report

Unlike some other examples' `push_step`, this example's pusher does not push
an `eval_report` artifact: `train_tabular()` returns a `ModelArtifact`
without a training-metrics dict, so there is nothing meaningful to report.

## Remote Run

Pass environment variables via `--environ` flags — they are serialized into the
Cadence/Temporal workflow and injected into every task's runtime environment,
reaching remote workers. Shell `export` statements before the command only
affect the local launcher and do not propagate.

```bash
python -m michelangelo_examples.california_housing.pipelines.pytorch_train.pipeline \
  remote-run \
  --image ghcr.io/michelangelo-ai/michelangelo-examples:california-housing \
  --storage-url s3://your-bucket/workflows \
  --environ AWS_ENDPOINT_URL=http://your-minio-endpoint:9000 \
  --environ AWS_ACCESS_KEY_ID=your-access-key \
  --environ AWS_SECRET_ACCESS_KEY=your-secret-key \
  --environ REGISTRY_ENDPOINT=your-apiserver-host:15566 \
  --yes
```

### k3d sandbox

```bash
python -m michelangelo_examples.california_housing.pipelines.pytorch_train.pipeline \
  remote-run \
  --image ghcr.io/michelangelo-ai/michelangelo-examples:california-housing \
  --storage-url s3://michelangelo/workflows \
  --environ AWS_ENDPOINT_URL=http://minio:9091 \
  --environ AWS_ACCESS_KEY_ID=minioadmin \
  --environ AWS_SECRET_ACCESS_KEY=minioadmin \
  --environ REGISTRY_ENDPOINT=michelangelo-apiserver:15566 \
  --yes
```

The `ghcr.io/michelangelo-ai/michelangelo-examples:california-housing` image is built by
this repo's own CI (`.github/workflows/build-image.yaml`) from the root
`Dockerfile` — no need to build it yourself unless testing a local change:

```bash
docker build -t michelangelo-examples:local .
k3d image import michelangelo-examples:local -c michelangelo-sandbox
kubectl delete cachedoutputs --all   # clear stale cached task outputs
```

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `AWS_ENDPOINT_URL` | No | — | S3-compatible endpoint URL (include scheme, e.g. `http://minio:9091`). Unset → local storage |
| `AWS_ACCESS_KEY_ID` | If `AWS_ENDPOINT_URL` set | — | Access key ID |
| `AWS_SECRET_ACCESS_KEY` | If `AWS_ENDPOINT_URL` set | — | Secret access key |
| `AWS_S3_BUCKET` | No | Parsed from `MA_FILE_SYSTEM` or `UF_STORAGE_URL` | Target bucket name |
| `REGISTRY_ENDPOINT` | No | — | Model registry gRPC endpoint (`host:port`). Unset → in-memory only |
| `REGISTRY_INSECURE` | No | `true` | Set `false` to enable TLS for the registry connection |
| `REGISTRY_NAMESPACE` | No | `MA_NAMESPACE` (the pipeline's own namespace), else `default` | Model registry namespace |

> **Sandbox note:** in a k3d sandbox, `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`,
> and `AWS_SECRET_ACCESS_KEY` are automatically injected into Ray/Spark pods
> via the `michelangelo-config` ConfigMap — no `--environ` flags needed for
> `ma pipeline run`. For `remote-run`, pass them explicitly with `--environ`.
> The apiserver Service and this example's task pods run in the same
> namespace by default, so `REGISTRY_ENDPOINT` can use the short in-cluster
> DNS name (`michelangelo-apiserver:15566`) rather than the full
> `michelangelo-apiserver.<namespace>.svc.cluster.local:15566` form.
