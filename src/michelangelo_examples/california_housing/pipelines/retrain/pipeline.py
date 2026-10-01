"""California Housing example: retrain composition, triggers pytorch-train as a child run."""

from __future__ import annotations

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.pipeline import run_pipeline

__all__ = ["retrain_workflow"]


@uniflow.workflow()
def retrain_workflow(
    namespace: str = "california-housing",
    pipeline_name: str = "pytorch-train",
    dataset_cols: str = (
        "MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,target"
    ),
    timeout_seconds: int = 3600,
    poll_seconds: int = 10,
):
    """Trigger the california_housing pytorch-train pipeline as a child run.

    Demonstrates the "one pipeline triggers another registered pipeline"
    composition pattern (michelangelo's Uniflow ``pipeline.run_pipeline``
    plugin) against a real, already-cron-triggered, already-model-registering
    pipeline, rather than a throwaway fixture pipeline.

    Args:
        namespace: Namespace of the child pipeline to trigger.
        pipeline_name: Name of the child pipeline to trigger.
        dataset_cols: Comma-separated column names forwarded to
            pytorch-train's own ``train_workflow(dataset_cols=...)`` kwarg.
        timeout_seconds: Max time to wait for the child run to finish.
        poll_seconds: Polling interval while waiting.

    Returns:
        The completed child ``PipelineRun`` dict returned by ``run_pipeline``
        (``metadata.name``, ``status.state``).
    """
    # TODO(https://github.com/michelangelo-ai/michelangelo-examples/issues/75):
    # for a true end-to-end retrain, resolve the model produced by this run
    # (model_search_by_pipeline_run, not yet a plugin) and deploy it via the
    # existing deployment plugin.
    return run_pipeline(
        namespace=namespace,
        pipeline_name=pipeline_name,
        kwargs={"dataset_cols": dataset_cols},
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )


if __name__ == "__main__":
    # No local runner: run_pipeline requires a live APIClient.PipelineRunService
    # (a real cluster/sandbox), unlike pytorch_train/xgb_train's `python -m`
    # local smoke path. Invoke via a registered PipelineRun instead:
    #   ma pipeline run --namespace california-housing --pipeline retrain
    raise SystemExit(
        "retrain_workflow requires a running Michelangelo sandbox; there is "
        "no local dry-run path (run_pipeline needs a live PipelineRunService). "
        "See README.md for how to trigger it against a sandbox."
    )
