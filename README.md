# michelangelo-examples

[![Test](https://github.com/michelangelo-ai/michelangelo-examples/actions/workflows/test.yaml/badge.svg)](https://github.com/michelangelo-ai/michelangelo-examples/actions/workflows/test.yaml)
[![Build examples Docker image](https://github.com/michelangelo-ai/michelangelo-examples/actions/workflows/build-image.yaml/badge.svg)](https://github.com/michelangelo-ai/michelangelo-examples/actions/workflows/build-image.yaml)

Pip-installable, ready-to-run example models and pipelines for
[Michelangelo](https://github.com/michelangelo-ai/michelangelo) — a
lightweight on-ramp for trying Michelangelo without a full monorepo
checkout, the heavyweight `example` extra, or a running sandbox.

```bash
pip install michelangelo-examples[<example-name>]
```

## Why this repo

The full example pipelines in
[`michelangelo/python/examples/`](https://github.com/michelangelo-ai/michelangelo/tree/main/python/examples)
are the canonical, full-fidelity reference — they exercise real
Cadence/Temporal orchestration, Spark, and a live sandbox, and they aren't
going anywhere. But trying one today means checking out the whole monorepo
and installing one heavyweight extra that bundles every example's
dependencies (`torch`, `transformers`, `xgboost`, `pyspark`, `ray`,
`mlflow`, `peft`, and more) at once, regardless of which example you
actually want.

`michelangelo-examples` is a second, lighter tier: each example is its own
pip extra with only the dependencies it needs, plus a plain-Python
`run_local.py` entrypoint that trains/evaluates locally — no Cadence, no
Spark, no sandbox. It's the "try it in five minutes" path underneath the
full pipeline reference, not a replacement for it.

## Examples

Ported so far:
[`california_housing`](src/michelangelo_examples/california_housing/) — a
project (use case) with two pipelines,
[`pytorch_train`](src/michelangelo_examples/california_housing/pipelines/pytorch_train/)
(PyTorch Lightning, migrated from core `michelangelo`'s
`python/examples/pipelines/california_housing_lightning/`) and
[`xgb_train`](src/michelangelo_examples/california_housing/pipelines/xgb_train/)
(distributed XGBoost via Ray Train). Structuring it as a project with a
`pipelines/` subfolder is what let the second pipeline slot in alongside
the first without a top-level rename.

v1 candidates (not yet ported): `movielens`, `bert-cola`. See the
[project spec](https://github.com/michelangelo-ai/michelangelo/tree/main/python/examples)
for the full list this repo is drawing from.

Each project lives entirely under
`src/michelangelo_examples/<project-name>/` — a real subpackage of the one
`michelangelo_examples` package this repo ships, including its
`pipeline.yaml`/`README.md` (bundled as package data so a plain `pip
install` gets everything, not just the Python code) alongside each
pipeline's model/training/task code and `__main__.py` local runner. Code
shared across a project's sibling pipelines (e.g. dataset loading/feature
prep) lives in `<project-name>/pipelines/libs/`.

## Installing a project

```bash
pip install "michelangelo-examples[california-housing]"
python -m michelangelo_examples.california_housing.pipelines.pytorch_train
```

Extras are scoped per **project**, not per pipeline: installing
`california-housing` pulls in the dependencies for every pipeline under
that project (`pytorch_train` and `xgb_train`), since they already
share one built image and mostly overlapping dependency sets.

The `pip install` + `python -m` commands above run the lightweight local
tier only, and only `pytorch_train` has one — `xgb_train` trains via
Ray Train's `XGBoostTrainer` from the start and always requires a sandbox.
To run a pipeline's full Cadence-dispatched version against a
Michelangelo sandbox (`ma project apply` → `ma pipeline apply` →
`ma pipeline run`), see the end-to-end command sequence in that pipeline's
own README — e.g.
[`pytorch_train`'s](src/michelangelo_examples/california_housing/pipelines/pytorch_train/README.md#end-to-end-sandbox-to-running-pipeline).

## Relationship to other Michelangelo repos

- [`michelangelo`](https://github.com/michelangelo-ai/michelangelo) — the
  core platform; `python/examples/` there remains the full pipeline
  reference (Cadence/Temporal/Spark/sandbox).
- [`michelangelo-models`](https://github.com/michelangelo-ai/michelangelo-models) —
  pluggable model architecture definitions. Some examples here may import
  a seed architecture from there instead of defining their own model.

## Development

This repo uses [`uv`](https://docs.astral.sh/uv/) for dependency
management.

```bash
uv build
uv run pytest
```

## License

[Apache License 2.0](LICENSE)
