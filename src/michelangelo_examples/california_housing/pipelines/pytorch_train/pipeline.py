"""PyTorch Lightning regression workflow for California Housing price prediction.

Workflow entry point that orchestrates the full California Housing pipeline
via ``tabular_trainer``'s Lightning backend: feature preparation, Spark
preprocessing, distributed Lightning training with Ray Train, an assembler
step that packages the trained model into deployable/raw Triton packages,
and a pusher step that exports the packaged model and preprocessed datasets
to storage and registry.
"""

from __future__ import annotations

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import RayTask
from michelangelo.uniflow.plugins.spark import SparkTask

from michelangelo_examples.california_housing.pipelines.libs.tasks.feature_prep import (
    feature_prep,
)
from michelangelo_examples.california_housing.pipelines.libs.tasks.preprocess import (
    PreprocessResult,
    preprocess,
)
from michelangelo_examples.california_housing.pipelines.pytorch_train.assembler import (
    assembler,
)
from michelangelo_examples.california_housing.pipelines.pytorch_train.push import (
    push_step,
)
from michelangelo_examples.california_housing.pipelines.pytorch_train.train import train

__all__ = [
    "PreprocessResult",
    "assembler",
    "feature_prep",
    "preprocess",
    "push_step",
    "train",
    "train_workflow",
]

# California Housing features + target column order.
# MedHouseVal (the sklearn target) is renamed to "target" in feature_prep.


@uniflow.workflow()
def train_workflow(
    dataset_cols: str = (
        "MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,target"
    ),
):
    """End-to-end ML workflow: feature prep, preprocessing, training, and push.

    Orchestrates the full ML lifecycle for California Housing using
    ``tabular_trainer``'s Lightning backend: feature preparation,
    preprocessing with Spark, distributed Lightning training with Ray Train,
    and a pusher step that pushes the trained model and preprocessed
    datasets to storage and registry.

    Args:
        dataset_cols: Comma-separated string of column names including
            features and target.

    Returns:
        List of PusherResult from push_step, one per artifact pushed.
    """
    _dataset_cols = dataset_cols.split(",")
    feature_prep_overrides = feature_prep.with_overrides(
        alias="feature_prep_overrides",
        config=RayTask(
            head_cpu=2,
            worker_instances=1,
        ),
    )
    train_dv, validation_dv = feature_prep_overrides(
        columns=_dataset_cols,
    )
    pr = preprocess.with_overrides(
        alias="preprocess_overrides",
        config=SparkTask(executor_cpu=1, driver_cpu=1),
    )(
        cast_float_columns=_dataset_cols,
        train_dv=train_dv,
        validation_dv=validation_dv,
    )
    model_artifact = train(
        pr,
        feature_columns=_dataset_cols[:-1],
    )
    assembled = assembler(model_artifact)
    return push_step(pr, assembled)


if __name__ == "__main__":
    ctx = uniflow.create_context()

    ctx.environ["IMAGE_PULL_POLICY"] = "IfNotPresent"

    # Pass MINIO_* and REGISTRY_* via --environ flags on the command line
    # so values reach remote Ray workers (see README Remote Run section).

    ctx.run(train_workflow)
