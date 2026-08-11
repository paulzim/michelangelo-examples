"""XGBoost regression workflow for California Housing price prediction.

Workflow entry point that orchestrates the full California Housing pipeline:
feature preparation, Spark preprocessing, distributed XGBoost training with
Ray, and a single pusher step that exports the model, evaluation report, and
preprocessed datasets to storage and registry.
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
from michelangelo_examples.california_housing.pipelines.xgb_train.push import push_step
from michelangelo_examples.california_housing.pipelines.xgb_train.train import (
    TrainResult,
    train,
)

__all__ = [
    "PreprocessResult",
    "TrainResult",
    "feature_prep",
    "preprocess",
    "push_step",
    "train",
    "train_workflow",
]


@uniflow.workflow()
def train_workflow(
    dataset_cols: str = (
        "MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,target"
    ),
):
    """End-to-end ML workflow: feature prep, preprocessing, training, and push.

    Orchestrates the full ML lifecycle for California Housing: feature
    preparation, preprocessing with Spark, distributed training with Ray
    XGBoost, and a single pusher step that pushes the trained model, evaluation
    report, and preprocessed datasets to storage and registry.

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
    train_result = train(
        pr,
        params={
            "objective": "reg:squarederror",
            "colsample_bytree": 0.3,
            "learning_rate": 0.1,
            "max_depth": 5,
            "alpha": 10,
            "n_estimators": 10,
        },
    )
    return push_step(pr, train_result)


if __name__ == "__main__":
    ctx = uniflow.create_context()

    ctx.environ["IMAGE_PULL_POLICY"] = "IfNotPresent"

    ctx.run(train_workflow)
