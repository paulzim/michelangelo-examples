"""Local, plain-Python entrypoint for the California Housing Lightning example.

No Ray, no Spark, no Cadence, no Michelangelo sandbox required -- this is
the lightweight tier described in the michelangelo-examples README: a fast
way to see ``TorchRegressionModel`` train and evaluate on a laptop. For the
full, production-shaped pipeline (feature_prep -> preprocess -> train ->
push_step, dispatched through Cadence via Uniflow), see ``pipeline.py`` and
``pipeline.yaml`` in this same package.

Usage:
    python -m michelangelo_examples.california_housing.pipelines.pytorch_train
"""

from __future__ import annotations

import logging

import pytorch_lightning as pl
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from michelangelo_examples.california_housing.pipelines.pytorch_train.model import (
    TorchRegressionModel,
)

log = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
LABEL_COLUMN = "target"


class _CaliforniaHousingDataset(Dataset):
    """Wraps a pandas DataFrame as a dict-per-sample torch Dataset.

    ``TorchRegressionModel.training_step``/``validation_step`` expect each
    batch to be a dict of column name -> tensor (the same shape Ray Data's
    default ``iter_torch_batches`` produces for the full pipeline) --
    ``DataLoader``'s default collate function produces exactly that shape
    when each sample is itself a dict, so no custom collate_fn is needed.
    """

    def __init__(self, frame):
        self._frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self._frame.iloc[idx]
        return {col: torch.tensor(row[col], dtype=torch.float32) for col in row.index}


def main(
    max_epochs: int = 5,
    sample_size: int = 2000,
    batch_size: int = 64,
    test_size: float = 0.25,
    seed: int = 1,
) -> TorchRegressionModel:
    """Train TorchRegressionModel locally on a small slice of California Housing.

    Args:
        max_epochs: Number of training epochs.
        sample_size: Number of rows to sample from the full ~20k-row dataset,
            for a fast local run. Set to 0 to use the full dataset.
        batch_size: DataLoader batch size.
        test_size: Fraction of the sampled data held out for validation.
        seed: Random seed for the sample/split and model init.

    Returns:
        The trained ``TorchRegressionModel``.
    """
    pl.seed_everything(seed)

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.rename(columns={"MedHouseVal": LABEL_COLUMN})
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=seed)

    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed)

    train_loader = DataLoader(
        _CaliforniaHousingDataset(train_df), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        _CaliforniaHousingDataset(val_df), batch_size=batch_size
    )

    model = TorchRegressionModel(
        feature_columns=FEATURE_COLUMNS,
        label_column=LABEL_COLUMN,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    log.info(
        "Trained TorchRegressionModel for %d epochs on %d rows (val loss: %s)",
        max_epochs,
        len(df),
        trainer.callback_metrics.get("val_loss"),
    )
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
