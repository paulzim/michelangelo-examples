"""GPT-OSS-20B fine-tuning workflow: dataset prep, LoRA training, evaluation.

Ported from michelangelo core's
``python/examples/gpt_oss_20b_finetune/simple_workflow.py``, restructured to
follow ``michelangelo_examples.bert_cola.pipelines.train.pipeline``'s shape
(module-level ``@uniflow.workflow()`` importing sibling task modules from
this package instead of core's ``examples.gpt_oss_20b_finetune``).
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    raise RuntimeError(
        "gpt-oss-20b-finetune requires Python 3.11+; this interpreter is "
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )

import os

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import UF_PLUGIN_RAY_USE_FSSPEC

from michelangelo_examples.gpt_oss_20b_finetune.pipelines.train.data import (
    prepare_finetune_dataset,
)
from michelangelo_examples.gpt_oss_20b_finetune.pipelines.train.eval import (
    evaluate_gpt_model,
)
from michelangelo_examples.gpt_oss_20b_finetune.pipelines.train.train import (
    simple_train_gpt,
)

__all__ = [
    "evaluate_gpt_model",
    "prepare_finetune_dataset",
    "simple_train_gpt",
    "train_workflow",
]


@uniflow.workflow()
def train_workflow(
    dataset_name="alpaca", num_epochs=1, sample_size=100, model_name="gpt2"
):
    """LoRA fine-tuning workflow: prepare data, train, evaluate.

    Args:
        dataset_name: One of "alpaca", "dolly", "oasst1".
        num_epochs: Number of training epochs.
        sample_size: Number of dataset samples to use.
        model_name: Base causal-LM checkpoint (e.g. "gpt2").
    """
    train_dv, val_dv, test_dv = prepare_finetune_dataset(
        dataset_name=dataset_name,
        max_length=512,
        sample_size=sample_size,
        model_name=model_name,
    )

    train_result = simple_train_gpt(
        train_dv=train_dv,
        validation_dv=val_dv,
        model_name=model_name,
        num_epochs=num_epochs,
        batch_size=1,
        learning_rate=5e-5,
        use_lora=True,
    )

    evaluate_gpt_model(
        test_dv=test_dv,
        checkpoint_path=train_result["checkpoint_path"],
        model_name=model_name,
        use_lora=True,
        lora_rank=16,
        learning_rate=5e-5,
        max_length=512,
        batch_size=1,
        num_samples=3,
    )

    return True


if __name__ == "__main__":
    ctx = uniflow.create_context()

    ctx.environ["DATA_SIZE"] = "100"
    ctx.environ[UF_PLUGIN_RAY_USE_FSSPEC] = "0"
    ctx.environ["MA_NAMESPACE"] = "default"
    ctx.environ["IMAGE_PULL_POLICY"] = "IfNotPresent"
    ctx.environ["S3_ALLOW_BUCKET_CREATION"] = "True"
    os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"

    if ctx.is_local_run():
        ctx.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
        ctx.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9091"
        ctx.environ["MA_API_SERVER"] = "localhost:15566"
    else:
        ctx.environ["MLFLOW_TRACKING_URI"] = "http://mlflow-proxy:5001"
        ctx.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9091"
        ctx.environ["MA_API_SERVER"] = "michelangelo-apiserver:15566"

    ctx.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = "s3://mlflow"
    ctx.environ["AWS_ACCESS_KEY_ID"] = "michelangeloadmin"
    ctx.environ["AWS_SECRET_ACCESS_KEY"] = "michelangeloadmin"

    ctx.run(
        train_workflow,
        dataset_name="alpaca",
        num_epochs=3,
        sample_size=50,
        model_name="gpt2",
    )
