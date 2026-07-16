# california_housing

A project (use case): predicting California housing prices. Pipelines
under this project share one dependency set
(`pip install "michelangelo-examples[california-housing]"`) and one built
image (`ghcr.io/michelangelo-ai/michelangelo-examples:california-housing`),
applied together via this project's own
[`config/project.yaml`](config/project.yaml)
(`ma project apply -f src/michelangelo_examples/california_housing/config/project.yaml`).

## Pipelines

- [`pytorch_train`](pipelines/pytorch_train/) —
  PyTorch Lightning regression via `tabular_trainer`, migrated from core
  `michelangelo`'s
  `python/examples/pipelines/california_housing_lightning/`. See its own
  README for how to run it locally or against a Michelangelo sandbox.
- [`xgboost_train`](pipelines/xgboost_train/) —
  distributed XGBoost regression via Ray Train's `XGBoostTrainer`, migrated
  from a fork's `python/examples/pipelines/california_housing_xgb/`. No
  local-only quick start — unlike `pytorch_train`, training goes through
  Ray Train from the start, so it always requires a Michelangelo sandbox.
  See its own README for the sandbox command sequence.

Both pipelines share [`feature_prep`/`preprocess`](pipelines/libs/tasks/)
against the same prepared dataset.

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

