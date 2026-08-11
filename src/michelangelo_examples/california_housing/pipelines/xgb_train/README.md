# California Housing XGBoost

End-to-end ML pipeline for California Housing price prediction using XGBoost.
Demonstrates the full Michelangelo workflow: feature preparation, Spark
preprocessing, distributed Ray training, and a pusher step that exports the
model, evaluation report, and preprocessed datasets to storage and registry.

This is the XGBoost variant of the California Housing project. For the
PyTorch Lightning variant, see
[`pytorch_train/`](../pytorch_train/README.md). Both pipelines share the
same feature preparation and preprocessing steps via
[`libs/tasks/`](../libs/tasks/).

## Pipeline

```
feature_prep  ->  preprocess  ->  train  ->  push_step
   (Ray)           (Spark)       (Ray)      (Spark)
```

| Step | File | Runtime | Description |
|---|---|---|---|
| `feature_prep` | `../libs/tasks/feature_prep.py` | Ray | Load dataset, train/test split, Ray Datasets |
| `preprocess` | `../libs/tasks/preprocess.py` | Spark | Cast columns to float |
| `train` | `train.py` | Ray | Distributed XGBoost training with dynamic ScalingConfig |
| `push_step` | `push.py` | Spark | Push model, eval report, and preprocessed datasets to storage/registry |

The workflow is orchestrated in `pipeline.py`, which imports the shared
feature preparation and preprocessing tasks from `libs/tasks/` and the
XGBoost-specific training and push steps from this directory.

## Prerequisites

- A Michelangelo sandbox running (`ma sandbox create`)
- A project created: `ma project apply -f src/michelangelo_examples/california_housing/config/project.yaml`
- Python 3.10+
- Java 17 with `JAVA_HOME` set (required for Spark; Java 21 is incompatible
  with PySpark 3.5 + Hadoop 3.3)
- Dependencies installed: `uv sync --extra california-housing`

Uses the [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
from scikit-learn (20,640 samples, 8 features, median house value target).

## How It Works

### UniFlow decorators

Each step is a plain Python function decorated with `@uniflow.task`. The decorator
registers the function with a runtime config (`RayTask` or `SparkTask`):

```python
@uniflow.task(
    config=RayTask(head_cpu=1, worker_cpu=1, worker_instances=2),
)
def train(pr: PreprocessResult, params: dict) -> TrainResult:
    ...
```

`@uniflow.workflow` composes tasks into a DAG that UniFlow transpiles to
Starlark for deterministic execution on Cadence/Temporal:

```python
@uniflow.workflow()
def train_workflow(dataset_cols: str = ...):
    train_dv, validation_dv = feature_prep(columns=...)
    pr = preprocess(train_dv=train_dv, ...)
    train_result = train(pr, params={...})
    return push_step(pr, train_result)
```

### Dynamic ScalingConfig

Unlike `pytorch_train`'s fixed `worker_instances`, this pipeline's `train`
step dynamically sizes its `ScalingConfig` from `ray.cluster_resources()` at
runtime -- using all available cluster CPUs (reserving 50% for Ray Data) and
detecting GPU availability automatically.

### push_step -- single pusher for all artifacts

`push_step` receives both `PreprocessResult` (for the datasets) and `TrainResult`
(for the model checkpoint) and pushes four artifacts in one Spark task:

| Artifact | Plugin | Sink |
|---|---|---|
| `model` | `ModelPusherPlugin` | `StorageBackend` (MinIO or local) + registry |
| `eval_report` | `EvalReportPusherPlugin` | `StorageBackend` (MinIO or local) + registry |
| `train_data` | `DatasetPusherPlugin` | `S3Sink` (remote) / `LocalFileSink` (local/CI) |
| `validation_data` | `DatasetPusherPlugin` | `S3Sink` (remote) / `LocalFileSink` (local/CI) |

## Remote Run (k3d sandbox)

```bash
# Build and import the project image
docker build -t michelangelo-examples:local .
k3d image import michelangelo-examples:local -c michelangelo-sandbox

# Create the project (if not already created)
ma project apply -f src/michelangelo_examples/california_housing/config/project.yaml

# Apply and run the pipeline
ma pipeline apply -f src/michelangelo_examples/california_housing/pipelines/xgb_train/pipeline.yaml
ma pipeline run -n california-housing --name xgb-train
```

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `AWS_ENDPOINT_URL` | No | -- | S3-compatible endpoint URL (e.g. `http://minio:9091`). Unset = local storage |
| `AWS_ACCESS_KEY_ID` | If remote | -- | Access key ID |
| `AWS_SECRET_ACCESS_KEY` | If remote | -- | Secret access key |
| `AWS_S3_BUCKET` | No | Parsed from `MA_FILE_SYSTEM`/`UF_STORAGE_URL` | Target bucket name |
| `REGISTRY_ENDPOINT` | No | -- | Model registry gRPC endpoint. Unset = in-memory only |
| `REGISTRY_INSECURE` | No | `true` | Set `false` for TLS |
| `REGISTRY_NAMESPACE` | No | `MA_NAMESPACE` or `default` | Model registry namespace |

In a k3d sandbox, `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, and
`AWS_SECRET_ACCESS_KEY` are automatically injected into Ray/Spark pods
via the `michelangelo-config` ConfigMap.

## Expected Output

```
INFO  feature_prep  Train dataset schema: Schema(MedInc: double, HouseAge: double, ...)
INFO  preprocess    Processed Train Spark schema: ...
INFO  train         scaling_config: ScalingConfig(num_workers=1, ...)
INFO  train         run_config: RunConfig(storage_path='s3://.../ray_results')
INFO  push_step     Found model checkpoint: .../model.ubj
INFO  push_step     push model (model_plugin): success=True ...
INFO  push_step     push eval_report (eval_report_plugin): success=True ...
INFO  push_step     push train_data (dataset_plugin): success=True ...
INFO  push_step     push validation_data (dataset_plugin): success=True ...
```
