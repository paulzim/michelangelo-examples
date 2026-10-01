"""End-to-end MovieLens-100k training using the michelangelo Lightning trainer.

Ported from michelangelo core's ``python/examples/movielens/train.py``,
repointing the ``load_movielens_100k``/``create_ncf_model`` imports at this
project's own ``data.py``/``model.py`` instead of core's
``examples.movielens`` package.

Trains a tiny NCF on CPU with a single Ray Train worker -- the smallest
viable smoke test for
:class:`michelangelo.lib.trainer.torch.pytorch_lightning.lightning_trainer.LightningTrainer`.

Unlike core's version, this migrated copy drops the optional Comet/MLflow
experiment-tracking code paths entirely (core's Comet path called a core
platform utility, ``michelangelo.lib.trainer.torch.pytorch_lightning.
_private.util.build_comet_logger``, by dotted-path string -- not something
usable standalone outside core, and not worth an opt-in extra here). This
example always uses Lightning's default local logger.
"""

from __future__ import annotations

import logging
import os

import ray
from michelangelo.lib.trainer.torch.pytorch_lightning.lightning_trainer import (
    LightningTrainer,
    LightningTrainerParam,
)

from michelangelo_examples.movielens.pipelines.train.data import load_movielens_100k
from michelangelo_examples.movielens.pipelines.train.model import create_ncf_model

log = logging.getLogger(__name__)

_STORAGE_DIR = "/tmp/movielens_runs"


def main() -> dict:
    """Run the MovieLens-100k NCF training and return the summary dict."""
    splits = load_movielens_100k()

    lightning_trainer_kwargs = {
        "max_epochs": 3,
        "log_every_n_steps": 20,
        # Force CPU for this local smoke test.
        #
        # On Apple Silicon, the trainer's strategy auto-selection picks
        # RaySingleDeviceStrategy for a single worker, which itself resolves
        # to MPS whenever it's available (mirroring Lightning's own
        # auto-detection order: MPS > CUDA > CPU) -- regardless of this
        # accelerator="cpu" kwarg, which then conflicts with that
        # MPS-resolved strategy device and raises a
        # ``MisconfigurationException``. Explicitly requesting the "ddp"
        # strategy (which this trainer supports even for a single worker)
        # sidesteps that single-device MPS auto-detection entirely, so
        # accelerator="cpu" is honored on both macOS and Linux/CI.
        "accelerator": "cpu",
        "strategy": "ddp",
    }

    trainer_param = LightningTrainerParam(
        create_model_fn=create_ncf_model,
        create_model_fn_kwargs={
            "num_users": splits.num_users,
            "num_items": splits.num_items,
            "embedding_dim": 32,
            "hidden_dim": 64,
            "learning_rate": 1e-3,
        },
        train_data=splits.train,
        val_data=splits.val,
        batch_size=256,
        num_shuffle_batches=10,
        lightning_trainer_kwargs=lightning_trainer_kwargs,
    )

    os.makedirs(_STORAGE_DIR, exist_ok=True)
    run_config = ray.train.RunConfig(
        name="ncf_movielens100k",
        storage_path=_STORAGE_DIR,
    )
    scaling_config = ray.train.ScalingConfig(
        num_workers=1,
        use_gpu=False,
        resources_per_worker={"CPU": 1},
    )

    trainer = LightningTrainer(
        trainer_param=trainer_param,
        run_config=run_config,
        scaling_config=scaling_config,
    )

    log.info("Starting training...")
    result = trainer.train()
    log.info("Training finished. result=%r", result)
    return result
