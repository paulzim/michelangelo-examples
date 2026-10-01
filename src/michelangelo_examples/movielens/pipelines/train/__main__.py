"""Local entrypoint for the MovieLens-100k NCF example.

This project has no full uniflow ``pipeline.py``/``pipeline.yaml`` -- core's
``python/examples/movielens/`` has no uniflow workflow to port (it's a flat
``data.py``/``model.py``/``train.py`` invoked directly via
``python -m examples.movielens.train``), so this ``__main__.py`` is the
project's only entry point. Unlike this repo's other ``__main__.py``-only
local runners (e.g. ``california_housing``'s), this one is *not* a
simplified, Ray-free proxy -- exercising the real Ray-backed
``LightningTrainer`` end to end (single CPU worker) is the entire point of
this demo, so ``train.py``'s Ray Train + PyTorch Lightning flow is preserved
as-is.

Usage:
    python -m michelangelo_examples.movielens.pipelines.train
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    raise RuntimeError(
        "movielens requires Python 3.11+; this interpreter is "
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )

import logging

from michelangelo_examples.movielens.pipelines.train.train import main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
