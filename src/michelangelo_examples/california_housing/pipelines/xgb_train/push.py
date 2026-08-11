"""Pusher step for the California Housing XGBoost workflow.

Pushes all pipeline artifacts in a single Spark task: trained XGBoost model,
evaluation report, and preprocessed train/validation datasets. All four artifacts
share the same storage backend -- MinIO / S3-compatible for remote runs,
local filesystem for development and CI.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import michelangelo.uniflow.core as uniflow
from michelangelo.lib.model_manager.constants import ModelKind
from michelangelo.uniflow.plugins.spark import SparkTask
from michelangelo.workflow.schema.pusher import (
    DatasetPluginConfig,
    EvalReportPluginConfig,
    ModelPluginConfig,
    PusherConfig,
    PusherPluginConfig,
)
from michelangelo.workflow.tasks.pusher import push
from michelangelo.workflow.variables.types import (
    AssembledModel,
    ModelArtifact,
    PusherResult,
)

from michelangelo_examples.california_housing.pipelines.xgb_train._backend import (
    resolve_storage_backend,
)

if TYPE_CHECKING:
    from michelangelo_examples.california_housing.pipelines.libs.tasks.preprocess import (
        PreprocessResult,
    )
    from michelangelo_examples.california_housing.pipelines.xgb_train.train import (
        TrainResult,
    )

log = logging.getLogger(__name__)

__all__ = ["push_step"]


@uniflow.task(
    config=SparkTask(
        driver_cpu=1,
        driver_memory="4G",
        executor_cpu=1,
        executor_memory="2G",
        executor_instances=1,
    ),
)
def push_step(
    pr: PreprocessResult,
    train_result: TrainResult,
) -> list[PusherResult]:
    """Push all pipeline artifacts to storage and registry in a single Spark step.

    Pushes four artifacts using a single storage backend selected at runtime:

    - **model** -- trained XGBoost checkpoint via ``ModelPusherPlugin``.
    - **eval_report** -- training metrics via ``EvalReportPusherPlugin``.
    - **train_data** -- preprocessed training dataset via ``DatasetPusherPlugin``
      + ``S3Sink`` (remote) or ``LocalFileSink`` (local/CI).
    - **validation_data** -- preprocessed validation dataset via
      ``DatasetPusherPlugin`` + ``S3Sink`` (remote) or ``LocalFileSink`` (local/CI).

    Args:
        pr: Result of the ``preprocess`` task, holding preprocessed training
            and validation ``DatasetVariable`` handles.
        train_result: Result of the ``train`` task, holding the XGBoost
            checkpoint path and training metrics.

    Returns:
        List of ``PusherResult``, one per artifact pushed.
    """
    import os
    import tempfile

    import fsspec

    storage_backend, is_remote = resolve_storage_backend("california_xgb_push_")

    # ── Locate XGBoost checkpoint ────────────────────────────────────────────
    # train_result.path is a directory (Ray's XGBoostTrainer run directory),
    # local or s3://. fsspec's url_to_fs() picks the right filesystem
    # (LocalFileSystem or s3fs, using the same AWS_* env vars s3fs already
    # reads elsewhere in this pipeline) so both cases share one glob/get
    # call instead of branching between a raw Minio client and glob.glob().
    #
    # Ray's Result.path is deliberately scheme-less even for remote/cloud
    # storage (e.g. "default/ray_results/run-..." for storage_path
    # "s3://default/ray_results") -- Ray expects callers to pair it with
    # result.filesystem rather than treat it as a URI. Re-qualify it with
    # the scheme our own resolve_storage_backend() already determined,
    # otherwise fsspec.core.url_to_fs() defaults to LocalFileSystem and
    # looks for the checkpoint on the Spark driver's local disk, where it
    # was never written.
    raw_path = train_result.path
    if is_remote and "://" not in raw_path:
        raw_path = f"s3://{raw_path}"
    fs, fs_path = fsspec.core.url_to_fs(raw_path)
    matches = fs.glob(f"{fs_path}/**/model.ubj")
    if not matches:
        matches = [p for p in fs.glob(f"{fs_path}/**/*") if fs.isfile(p)]
    if not matches:
        raise FileNotFoundError(f"No model checkpoint found under {raw_path}")

    tmp_ckpt_dir = tempfile.mkdtemp(prefix="checkpoint_")
    checkpoint_path = os.path.join(tmp_ckpt_dir, "model.ubj")
    fs.get(matches[0], checkpoint_path)
    log.info("Found model checkpoint: %s", checkpoint_path)

    _run_id = os.path.basename(train_result.path)

    # ── Load datasets as pandas DataFrames ───────────────────────────────────
    pr.train_data.load_pandas_dataframe()
    pr.validation_data.load_pandas_dataframe()

    # ── Dataset sink config ──────────────────────────────────────────────────
    if is_remote:
        from michelangelo.workflow.schema.sinks.s3 import S3SinkConfig
        from michelangelo.workflow.tasks.functions.sinks import S3Sink

        def _dataset_config(key: str) -> DatasetPluginConfig:
            return DatasetPluginConfig(
                sinks=[S3Sink(S3SinkConfig(key, storage_backend=storage_backend))]
            )
    else:
        from michelangelo.workflow.schema.sinks.local import LocalFileSinkConfig
        from michelangelo.workflow.tasks.functions.sinks import LocalFileSink

        _local_dir = storage_backend.get_storage_location()

        def _dataset_config(key: str) -> DatasetPluginConfig:  # type: ignore[misc]
            return DatasetPluginConfig(
                sinks=[
                    LocalFileSink(
                        LocalFileSinkConfig(
                            destination_path=os.path.join(_local_dir, key)
                        )
                    )
                ]
            )

    # ── Registry client ───────────────────────────────────────────────────────
    registry_endpoint = os.environ.get("REGISTRY_ENDPOINT")
    if registry_endpoint:
        import grpc as _grpc
        from michelangelo.api.v2 import APIClient
        from michelangelo.lib.model_manager.registry.api_client import APIRegistryClient

        _insecure = os.environ.get("REGISTRY_INSECURE", "true").lower() != "false"
        _credentials = None if _insecure else _grpc.ssl_channel_credentials()
        _channel = (
            _grpc.insecure_channel(registry_endpoint)
            if _insecure
            else _grpc.secure_channel(registry_endpoint, _credentials)
        )
        _api_client = APIClient(
            caller="california-housing-xgb-push-step",
            channel=_channel,
        )
        registry_client = APIRegistryClient(
            svc=_api_client.ModelService,
            namespace=os.environ.get(
                "REGISTRY_NAMESPACE", os.environ.get("MA_NAMESPACE", "default")
            ),
        )
        log.info("push_step: using APIRegistryClient at %s", registry_endpoint)
    else:
        from michelangelo.lib.model_manager.registry.client import (
            InMemoryRegistryClient,
        )

        registry_client = InMemoryRegistryClient()
        log.warning(
            "REGISTRY_ENDPOINT not set -- using InMemoryRegistryClient. "
            "Model registration will not be persisted."
        )

    # ── Pusher config ─────────────────────────────────────────────────────────
    from michelangelo.gen.api.v2.evaluation_report_pb2 import (
        EvaluationReport,
        EvaluationReportSpec,
    )

    metrics = {k: round(v, 4) for k, v in (train_result.metrics or {}).items()}
    config = PusherConfig(
        items=[
            PusherPluginConfig(
                name="model",
                model_plugin=ModelPluginConfig(
                    description="XGBoost regression on California Housing dataset",
                    kind=ModelKind.REGRESSION,
                    labels={"framework": "xgboost"},
                    metadata=metrics,
                ),
            ),
            PusherPluginConfig(
                name="eval_report",
                eval_report_plugin=EvalReportPluginConfig(
                    extra_fields=metrics,
                ),
            ),
            PusherPluginConfig(
                name="train_data",
                dataset_plugin=_dataset_config(
                    f"datasets/california-housing/{_run_id}/train"
                ),
            ),
            PusherPluginConfig(
                name="validation_data",
                dataset_plugin=_dataset_config(
                    f"datasets/california-housing/{_run_id}/validation"
                ),
            ),
        ]
    )

    assembled = AssembledModel(raw_model=ModelArtifact(path=checkpoint_path))
    eval_report = EvaluationReport(
        spec=EvaluationReportSpec(title="California Housing XGBoost Evaluation")
    )

    results = push(
        config=config,
        artifacts={
            "model": assembled,
            "eval_report": eval_report,
            "train_data": pr.train_data,
            "validation_data": pr.validation_data,
        },
        storage_backend=storage_backend,
        registry_client=registry_client,
    )

    for r in results:
        log.info(
            "push %s (%s): success=%s value=%s error=%s",
            r.name,
            r.plugin,
            r.success,
            r.value,
            r.error,
        )

    return results
