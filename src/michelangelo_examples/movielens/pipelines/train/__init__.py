"""MovieLens-100k NCF training pipeline.

Intentionally contains no imports. Importing this package in a Ray task
container would eagerly load torch/pandas at module level, which individual
task containers may not need -- keep this file import-free so each Ray
task container only loads what its own task module needs.
"""
