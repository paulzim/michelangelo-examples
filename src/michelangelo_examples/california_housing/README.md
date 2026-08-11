# california_housing

A project (use case): predicting California housing prices. Pipelines
under this project share one dependency set
(`pip install "michelangelo-examples[california-housing]"`) and one built
image (`ghcr.io/michelangelo-ai/michelangelo-examples:california-housing`),
applied together via this project's own
[`config/project.yaml`](config/project.yaml)
(`ma project apply -f src/michelangelo_examples/california_housing/config/project.yaml`).

## Pipelines

- [`pytorch_train`](pipelines/pytorch_train/) --
  PyTorch Lightning regression via `tabular_trainer`. See its own
  README for how to run it locally or against a Michelangelo sandbox.
- [`xgb_train`](pipelines/xgb_train/) --
  XGBoost distributed regression via Ray's `XGBoostTrainer`. Pushes model,
  eval report, and preprocessed datasets in a single Spark push step.

Both pipelines share the same feature preparation and preprocessing steps
via [`libs/tasks/`](pipelines/libs/tasks/).

## Layout

- `config/project.yaml` — this project's Michelangelo Project CRD config.
- `pipelines/<pipeline-name>/` — one directory per pipeline: model,
  training, and pipeline-task code, a `__main__.py` local runner
  (`python -m michelangelo_examples.california_housing.pipelines.<pipeline-name>`),
  this pipeline's `pipeline.yaml`, and its own `README.md`. These ship as
  real package contents — `pip install` gets the code, `pipeline.yaml`,
  and `README.md` together (verified via `uv build` + inspecting the
  wheel), not just the `.py` files.
- `pipelines/libs/` — code shared across this project's sibling pipelines
  (e.g. dataset loading/feature-prep steps common to both a Lightning and
  an XGBoost training approach), so adding a second pipeline doesn't mean
  duplicating that logic.

