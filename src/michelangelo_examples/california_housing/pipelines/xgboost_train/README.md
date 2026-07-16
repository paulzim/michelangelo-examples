# California Housing: XGBoost Train

End-to-end ML pipeline for California Housing price prediction using
distributed XGBoost via Ray Train's `XGBoostTrainer`, running against a
released [`michelangelo`](https://pypi.org/project/michelangelo/) PyPI
package — no core `michelangelo` monorepo checkout required.

Part of the `california_housing` project (use case); sibling to
[`pytorch_train`](../pytorch_train/).

Unlike `pytorch_train`, this pipeline has no local, Ray-free quick-start —
XGBoost training here goes through Ray Train's distributed
`XGBoostTrainer` from the start, so the only way to run it is the full
pipeline below.

```
feature_prep  →  preprocess  →  train  →  push_step
   (Ray)           (Spark)      (Ray)      (Ray)
```

| Step | File | Runtime | Description |
|---|---|---|---|
| `feature_prep` | [`libs/tasks/feature_prep.py`](../libs/tasks/feature_prep.py) | Ray | Load dataset (bundled CSV, falls back to network fetch), train/test split, Ray Datasets |
| `preprocess` | [`libs/tasks/preprocess.py`](../libs/tasks/preprocess.py) | Spark | Cast columns to float |
| `train` | [`train.py`](train.py) | Ray | Distributed XGBoost training via Ray Train's `XGBoostTrainer` |
| `push_step` | [`push.py`](push.py) | Ray | Push model, evaluation report, and preprocessed datasets to storage/registry |

`feature_prep` and `preprocess` are shared with `pytorch_train` (see
[`libs/`](../libs/)) — both pipelines train against the same prepared
dataset.

## Prerequisites

- A Michelangelo sandbox running (`ma sandbox create`)
- The `california-housing` project applied from this project's own config:
  `ma project apply -f src/michelangelo_examples/california_housing/config/project.yaml`
- Python 3.10+

If you hit sandbox issues getting to this point (helm timeouts, zombie Ray
clusters, Cadence domain registration, MA Studio 415 errors, ...), see
[`docs/sandbox-troubleshooting.md`](../../../../../docs/sandbox-troubleshooting.md).

## End-to-end: sandbox to running pipeline

The full command sequence a first-time operator needs, in order — each
step depends on the one before it:

```bash
# 1. Create a sandbox (skip if you already have one running)
ma sandbox create

# 2. Register the california-housing project (namespace: california-housing)
ma project apply -f src/michelangelo_examples/california_housing/config/project.yaml

# 3. Register this pipeline (namespace: california-housing, name: xgboost-train)
ma pipeline apply -f src/michelangelo_examples/california_housing/pipelines/xgboost_train/pipeline.yaml

# 4. Run it
ma pipeline run -n california-housing --name xgboost-train
```

`ma pipeline run` dispatches through Cadence using the image already
declared in `pipeline.yaml`'s `michelangelo/uniflow-image` annotation
(`ghcr.io/michelangelo-ai/michelangelo-examples:california-housing`, built
by this repo's own CI, shared with `pytorch_train`) — no `--image`/`--environ`
flags needed for this path. Use `remote-run` (below) instead of
`ma pipeline apply` + `ma pipeline run` if you need to override the image or
pass environment variables without registering the pipeline first.

## How It Works

### Distributed training via Ray Train's `XGBoostTrainer`

`train.py` builds a `train_loop_per_worker` closure that materializes each
worker's Ray Data shard to a pandas `DataFrame`, converts it to an
`xgboost.DMatrix`, and calls `xgboost.train()` with
`RayTrainReportCallback()` so Ray Train can track progress. Worker count and
CPU allocation are computed dynamically at runtime from the live Ray
cluster's resources (`create_scaling_config()`), rather than hardcoded, so
the same code scales from a small sandbox to a larger cluster without
edits.

### Locating the XGBoost checkpoint

Ray Train's `XGBoostTrainer` writes the trained model as `model.ubj` under
the run's storage path. `push_step` needs to resolve that path to a local
file before wrapping it in a `ModelArtifact`: on a remote run, storage is
S3-compatible and `glob` can't traverse it, so `push_step` uses a MinIO
client to list and download `model.ubj`; on a local run, it's a plain
`glob` under the local storage path.

### Eval report

Unlike `pytorch_train`'s pusher, this pipeline's `push_step` also pushes an
`eval_report` artifact: `XGBoostTrainer`'s result includes a metrics dict
(e.g. validation RMSE), so there's something meaningful to report.

## Remote Run

Pass environment variables via `--environ` flags — they are serialized into the
Cadence/Temporal workflow and injected into every task's runtime environment,
reaching remote workers. Shell `export` statements before the command only
affect the local launcher and do not propagate.

```bash
python -m michelangelo_examples.california_housing.pipelines.xgboost_train.pipeline \
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
python -m michelangelo_examples.california_housing.pipelines.xgboost_train.pipeline \
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
