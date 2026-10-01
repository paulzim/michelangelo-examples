# California Housing: Retrain (triggers pytorch-train or xgb-train)

Demonstrates triggering an already-registered pipeline (`pytorch-train` or
`xgb-train`) as a child run from another pipeline's Uniflow workflow, via
`michelangelo.uniflow.plugins.pipeline.run_pipeline`.

Part of the `california_housing` project (use case); for the pipelines this
one can trigger, see [`pytorch_train/`](../pytorch_train/README.md) and
[`xgb_train/`](../xgb_train/README.md).

`retrain_workflow` takes `pipeline_name` as an input parameter (default
`"pytorch-train"`), so the same workflow can trigger either sibling
pipeline — `pipeline.yaml`'s `weekly-retrain` trigger declares one
`parametersMap` entry per target (`retrain-pytorch-train`,
`retrain-xgb-train`), each passing a different `pipeline_name`. Both target
pipelines share the same `dataset_cols` parameter shape, so it's forwarded
unchanged in both entries.

No local runner: `run_pipeline` requires a live `PipelineRunService` (a real
sandbox/cluster), unlike `pytorch_train`/`xgb_train`'s `python -m` local
smoke path.

## Prerequisites

- A Michelangelo sandbox running (`ma sandbox create`)
- The `california-housing` project applied from this project's own config:
  `ma project apply -f src/michelangelo_examples/california_housing/config/project.yaml`
- `pytorch-train` and/or `xgb-train` already registered, depending on which
  one you intend to trigger (see [`pytorch_train/README.md`](../pytorch_train/README.md)'s
  "End-to-end: sandbox to running pipeline" section; `xgb_train/README.md`
  follows the same pattern)
- This `retrain` pipeline itself registered:
  `ma pipeline apply -f src/michelangelo_examples/california_housing/pipelines/retrain/pipeline.yaml`

## Run

```bash
ma pipeline run -n california-housing --name retrain
```

Or start the `weekly-retrain` trigger declared in `pipeline.yaml` so it runs
on its own cron schedule (Monday 06:00 UTC), the same one-time way
`pytorch_train`'s own `daily-noon-run` trigger is started:

```bash
ma trigger_run create --namespace=california-housing --pipeline=retrain --trigger-name=weekly-retrain
```

## Future follow-up (out of scope here)

A fuller retrain-and-deploy example (model lookup via
`get_models_by_pipeline_run` + `create_or_update_deployment`/
`wait_for_deployment`) is possible since `pytorch-train`'s `push_step` does
register a discoverable model in non-local runs — left as a separable
future addition.
