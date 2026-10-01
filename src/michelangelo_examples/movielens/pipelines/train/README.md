# MovieLens-100k NCF example

Smallest viable smoke test for
`michelangelo.lib.trainer.torch.pytorch_lightning.lightning_trainer.LightningTrainer`.
Trains a tiny Neural Collaborative Filtering model on MovieLens-100k on CPU
with a single Ray Train worker.

Migrated from core `michelangelo`'s `python/examples/movielens/`. Unlike
core's version, this one drops the optional Comet/MLflow experiment-tracking
code paths entirely (see [Deviations from core](#deviations-from-core)
below) -- it always uses Lightning's default local logger.

## Run

Requires Python 3.11+ (see this project's own [README](../../README.md)).

```bash
pip install "michelangelo-examples[movielens]"
python -m michelangelo_examples.movielens.pipelines.train
```

The first invocation downloads the dataset (~5 MB) to `/tmp/movielens_data/`.
Checkpoints land in `/tmp/movielens_runs/ncf_movielens100k/`.

## What it exercises

- Loading `LightningTrainer` and `LightningTrainerParam` from
  `michelangelo`'s published wheel.
- The trainer's per-worker training loop (`_train_loop_per_worker`) for a
  non-trivial end-to-end Lightning fit, including epoch checkpointing via
  `RayTrainReportCallback`.
- Default Ray Data -> torch tensor collation (no custom `data_collate_fn`).
- Resolving the default `RayDDPStrategy` even when running with a single
  worker.

## Files

- `data.py` -- downloads MovieLens-100k, builds dense user/item index,
  returns Ray datasets.
- `model.py` -- `NCFLightningModule` (user + item embeddings -> 2-layer
  MLP -> sigmoid, MSE loss).
- `train.py` -- wires up `LightningTrainerParam`, `LightningTrainer`, and
  Ray `RunConfig` / `ScalingConfig`.
- `__main__.py` -- the local entrypoint (`python -m ...pipelines.train`);
  also carries the Python 3.11+ runtime guard.

## Deviations from core

Core's `train.py` supports optional Comet or MLflow experiment tracking,
gated by env vars. This migrated version drops both entirely:

- Core's Comet path built its `CometLogger` via a dotted-path factory,
  `michelangelo.lib.trainer.torch.pytorch_lightning._private.util.
  build_comet_logger` -- genuine core-platform code, not something
  meaningful to reproduce standalone in this repo.
- Keeping either path opt-in behind a rarely-used env var would have meant
  pulling `comet_ml`/`mlflow` into this project's base extras for a code
  path most users won't exercise; simpler to drop it and always use
  Lightning's default local logger.
