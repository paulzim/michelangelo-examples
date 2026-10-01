"""Distributed GPT fine-tuning task using Ray Train + PyTorch Lightning.

Ported from michelangelo core's
``python/examples/gpt_oss_20b_finetune/simple_train.py``, repointing the
``create_gpt_model`` import at this project's own ``model.py`` instead of
core's ``examples.gpt_oss_20b_finetune`` package.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import michelangelo.uniflow.core as uniflow
import mlflow
from michelangelo.lib.trainer.torch.pytorch_lightning.lightning_trainer import (
    LightningTrainer,
    LightningTrainerParam,
)
from michelangelo.uniflow.plugins.ray import RayTask, create_run_config
from michelangelo.workflow.variables import DatasetVariable
from pytorch_lightning.loggers import MLFlowLogger
from ray.train import CheckpointConfig, ScalingConfig
from ray.train.lightning import RayFSDPStrategy

from michelangelo_examples.gpt_oss_20b_finetune.pipelines.train.model import (
    create_gpt_model,
)

if TYPE_CHECKING:
    from ray.data import Dataset

log = logging.getLogger(__name__)

__all__ = ["log_checkpoint_to_mlflow", "simple_train_gpt"]


def log_checkpoint_to_mlflow(checkpoint_path: str, run_id: str) -> str:
    """Log a checkpoint to MLflow artifacts (persisted to S3 automatically).

    Args:
        checkpoint_path: Local path to the checkpoint directory or file.
        run_id: MLflow run ID to log artifacts to.

    Returns:
        MLflow artifact URI for the checkpoint.
    """
    log.info("Logging checkpoint to MLflow artifacts: %s", checkpoint_path)
    log.info("Using MLflow run ID: %s", run_id)

    if os.path.isdir(checkpoint_path):
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifacts(checkpoint_path, "checkpoint")
        artifact_path = "checkpoint"
    else:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(checkpoint_path, "checkpoint")
        artifact_path = f"checkpoint/{os.path.basename(checkpoint_path)}"

    artifact_uri = f"runs:/{run_id}/{artifact_path}"
    log.info("Checkpoint logged to MLflow: %s", artifact_uri)
    return artifact_uri


@uniflow.task(
    config=RayTask(
        head_cpu=2,
        head_memory="8Gi",
        worker_cpu=1,
        worker_memory="4Gi",
        worker_instances=1,
    ),
    cache_enabled=False,
)
def simple_train_gpt(
    train_dv: DatasetVariable,
    validation_dv: DatasetVariable,
    model_name: str = "gpt2",
    num_epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 5e-5,
    use_lora: bool = True,
    lora_rank: int = 16,
    num_workers: int = 1,
    use_gpu: bool = True,
):
    """Fine-tune a GPT causal-LM checkpoint via Ray Train + Lightning + LoRA.

    Args:
        train_dv: Training dataset variable.
        validation_dv: Validation dataset variable.
        model_name: Base model name (e.g., "gpt2").
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate for optimization.
        use_lora: Whether to use LoRA (Low-Rank Adaptation).
        lora_rank: LoRA rank for parameter-efficient fine-tuning.
        num_workers: Number of worker nodes for distributed training.
        use_gpu: Whether to use GPU acceleration.

    Returns:
        Dict with the checkpoint's S3 URI and the MLflow run ID.
    """
    log.info("Starting distributed training with model: %s", model_name)
    log.info("Training with %d workers, use_gpu: %s", num_workers, use_gpu)

    train_dv.load_ray_dataset()
    train_data: Dataset = train_dv.value

    validation_dv.load_ray_dataset()
    validation_data: Dataset = validation_dv.value

    import torch

    if torch.backends.mps.is_available():
        # Apple Silicon - use CPU to avoid MPS/FSDP conflicts.
        use_gpu = False
        log.info("Detected Apple Silicon (MPS) - using CPU for compatibility")
    elif not torch.cuda.is_available():
        use_gpu = False
        log.info("No CUDA available - using CPU")

    scaling_config = ScalingConfig(
        trainer_resources={"CPU": 1},
        resources_per_worker={"CPU": 2},
        num_workers=num_workers,
        use_gpu=use_gpu,
    )
    log.info("scaling_config: %r", scaling_config)

    # Use the same S3 bucket that workers already have access to (s3://default),
    # avoiding permission issues with creating separate buckets.
    storage_path = "s3://default/ray_checkpoints"
    run_config = create_run_config(
        name=f"gpt-distributed-{model_name.replace('/', '-')}",
        storage_path=storage_path,
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
    )
    log.info("Using %s for Ray checkpoint storage", storage_path)
    log.info("run_config: %r", run_config)

    experiment_name = "gpt-finetune-experiment"
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"),
        run_name=f"training-{model_name}",
        tags={
            "model_name": model_name,
            "use_lora": str(use_lora),
            "lora_rank": str(lora_rank),
            "training_type": "distributed",
        },
    )

    mlflow_logger.log_hyperparams(
        {
            "model_name": model_name,
            "learning_rate": learning_rate,
            "use_lora": use_lora,
            "lora_rank": lora_rank,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
        }
    )

    if torch.backends.mps.is_available():
        lightning_trainer_kwargs = {
            "accelerator": "cpu",
            "precision": 32,  # MPS doesn't support mixed precision well.
            "log_every_n_steps": 10,
            "val_check_interval": 0.25,
            "logger": mlflow_logger,
        }
    elif use_gpu and torch.cuda.is_available():
        lightning_trainer_kwargs = {
            "strategy": RayFSDPStrategy(
                sharding_strategy="SHARD_GRAD_OP",
            ),
            "precision": "16-mixed",
            "log_every_n_steps": 10,
            "val_check_interval": 0.25,
            "logger": mlflow_logger,
        }
    else:
        lightning_trainer_kwargs = {
            "accelerator": "cpu",
            "precision": 32,
            "log_every_n_steps": 10,
            "val_check_interval": 0.25,
            "logger": mlflow_logger,
        }

    trainer_param = LightningTrainerParam(
        create_model_fn=create_gpt_model,
        create_model_fn_kwargs={
            "model_name": model_name,
            "learning_rate": learning_rate,
            "use_lora": use_lora,
            "lora_rank": lora_rank,
        },
        train_data=train_data,
        val_data=validation_data,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lightning_trainer_kwargs=lightning_trainer_kwargs,
    )

    trainer = LightningTrainer(trainer_param)

    log.info("Model configuration:")
    log.info("  Model name: %s", model_name)
    log.info("  Learning rate: %s", learning_rate)
    log.info("  Use LoRA: %s", use_lora)
    log.info("  LoRA rank: %s", lora_rank)
    log.info("  Batch size: %s", batch_size)
    log.info("  Number of epochs: %s", num_epochs)

    log.info("Starting distributed Lightning training...")
    result = trainer.train(run_config, scaling_config)
    log.info("Distributed training completed successfully")

    run_id = mlflow_logger.run_id
    log.info("Training completed, MLflow run: %s", run_id)

    checkpoint = result["checkpoint_path"]
    if hasattr(checkpoint, "as_directory"):
        checkpoint_path = checkpoint.path
    else:
        checkpoint_path = checkpoint

    log.info("Raw checkpoint path from Ray: %s", checkpoint_path)

    if not checkpoint_path.startswith("s3://"):
        checkpoint_path = f"s3://{checkpoint_path}"
        log.info("Corrected checkpoint path for S3: %s", checkpoint_path)

    log.info("Checkpoint already saved to S3, using S3 URI directly")

    return {
        "checkpoint_path": checkpoint_path,
        "mlflow_run_id": run_id,
    }
