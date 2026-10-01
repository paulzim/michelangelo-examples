# movielens

A project (use case): the smallest viable smoke test for
`michelangelo.lib.trainer.torch.pytorch_lightning.lightning_trainer.LightningTrainer`
-- a tiny Neural Collaborative Filtering (NCF) model trained on
MovieLens-100k, CPU-only, with a single Ray Train worker. Pipelines under
this project share one dependency set (`pip install
"michelangelo-examples[movielens]"`) and one built image
(`ghcr.io/michelangelo-ai/michelangelo-examples:movielens`), applied
together via this project's own
[`config/project.yaml`](config/project.yaml)
(`ma project apply -f src/michelangelo_examples/movielens/config/project.yaml`).

## Requires Python 3.11+

Unlike this repo's other projects (`bert-cola`, `california-housing`,
still on the package-wide `>=3.10` floor), this project's entry point
(`pipelines/train/__main__.py`) raises a `RuntimeError` at import time on
Python <3.11. This is a project-level requirement, not a `pip install`-time
restriction -- the top-level `michelangelo-examples` package still declares
`requires-python = ">=3.10"` so installing `bert-cola`/`california-housing`
on 3.10 is unaffected.

## Pipelines

- [`train`](pipelines/train/) -- downloads MovieLens-100k, builds a tiny
  NCF model, and trains it via Ray Train + PyTorch Lightning. No
  `pipeline.py`/`pipeline.yaml` -- migrated from core `michelangelo`'s
  `python/examples/movielens/`, which has no uniflow workflow to port, only
  a flat script invoked directly. See its own README for how to run it
  locally.

## Layout

- `config/project.yaml` -- this project's Michelangelo Project CRD config.
- `pipelines/train/` -- data, model, and training code, plus a
  `__main__.py` local runner and its own `README.md`. These ship as real
  package contents -- `pip install` gets the code and `README.md` together,
  not just the `.py` files.
