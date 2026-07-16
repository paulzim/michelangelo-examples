"""Feature preparation task for the California Housing project's pipelines.

Loads the California Housing dataset, performs a train/test split, and
converts the result to Ray Datasets for distributed processing. Shared by
every pipeline under this project (``pytorch_train``, ``xgboost_train``, ...).
"""

from __future__ import annotations

import logging
from pathlib import Path

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import RayTask
from michelangelo.workflow.variables import DatasetVariable

log = logging.getLogger(__name__)

__all__ = ["feature_prep"]

# Bundled dataset avoids a live network download at task runtime -- useful in
# sandboxes where an MITM-proxy CA cert (e.g. Zscaler) present on a
# developer's host isn't also present inside Ray worker containers, which
# then fail to fetch from an HTTPS-egressed source. Falls back to
# scikit-learn when absent, so this is backward compatible with any existing
# usage that relies on the network fetch.
_BUNDLED_DATA = Path(__file__).parent.parent / "data" / "california_housing.csv"


@uniflow.task(
    config=RayTask(
        head_cpu=1,
        head_gpu=0,
        head_memory="4Gi",
        worker_cpu=1,
        worker_gpu=0,
        worker_memory="4Gi",
        worker_instances=0,
    ),
    cache_enabled=False,  # off for tutorial simplicity; enable in production
)
def feature_prep(
    columns: list[str],
    test_size: float = 0.25,
    seed: int = 1,
) -> tuple[DatasetVariable, DatasetVariable]:
    """Prepare features from the California Housing dataset.

    Prefers a bundled CSV at ``pipelines/libs/data/california_housing.csv``
    to avoid network calls inside environments without egress (e.g. k3d
    sandboxes). Falls back to ``sklearn.datasets.fetch_california_housing``
    when the bundled file is absent.

    Args:
        columns: List of column names to select (features + ``"target"``).
        test_size: Fraction of data to use for validation. Defaults to 0.25.
        seed: Random seed for reproducibility. Defaults to 1.

    Returns:
        Tuple of (train_dataset, validation_dataset) as DatasetVariables.
    """
    import pandas as pd
    import ray.data

    if _BUNDLED_DATA.exists():
        log.info("Loading California Housing data from bundled CSV: %s", _BUNDLED_DATA)
        df = pd.read_csv(_BUNDLED_DATA)
    else:
        log.info(
            "Bundled CSV not found; downloading via sklearn (requires network access)"
        )
        from sklearn.datasets import fetch_california_housing

        housing = fetch_california_housing(as_frame=True)
        df = housing.frame.rename(columns={"MedHouseVal": "target"})

    data = ray.data.from_pandas(df).select_columns(columns)

    train_data, validation_data = data.train_test_split(
        test_size=test_size, shuffle=True, seed=seed
    )

    train_dv = DatasetVariable.create(train_data)
    train_dv.save_ray_dataset()

    validation_dv = DatasetVariable.create(validation_data)
    validation_dv.save_ray_dataset()

    log.info("Train dataset schema: %s", train_data.schema())
    log.info("Train dataset sample: %s", train_data.take(1))

    return train_dv, validation_dv
