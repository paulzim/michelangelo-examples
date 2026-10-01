# gpt_oss_20b_finetune

A project (use case): parameter-efficient (LoRA) fine-tuning of GPT-style
causal language models, architected for OpenAI's GPT-OSS-20B on a
GPU-backed cluster but tested against GPT-2 as a CPU-feasible proxy.
Pipelines under this project share one dependency set (`pip install
"michelangelo-examples[gpt-oss-20b-finetune]"`) and one built image
(`ghcr.io/michelangelo-ai/michelangelo-examples:gpt-oss-20b-finetune`),
applied together via this project's own
[`config/project.yaml`](config/project.yaml)
(`ma project apply -f src/michelangelo_examples/gpt_oss_20b_finetune/config/project.yaml`).

## Requires Python 3.11+

Unlike this repo's other projects (`bert-cola`, `california-housing`,
still on the package-wide `>=3.10` floor), this project's entry points
(`pipelines/train/__main__.py` and `pipelines/train/pipeline.py`) raise a
`RuntimeError` at import time on Python <3.11. This is a project-level
requirement, not a `pip install`-time restriction -- the top-level
`michelangelo-examples` package still declares `requires-python = ">=3.10"`
so installing `bert-cola`/`california-housing` on 3.10 is unaffected.

## Pipelines

- [`train`](pipelines/train/) -- prepares an instruction-following dataset
  (Alpaca by default), LoRA fine-tunes a causal-LM checkpoint via Ray Train
  + PyTorch Lightning, and evaluates perplexity/generation quality on a
  held-out split. See its own README for how to run it locally or against a
  Michelangelo sandbox.

## Layout

- `config/project.yaml` -- this project's Michelangelo Project CRD config.
- `pipelines/train/` -- model, data, training, and evaluation code, this
  pipeline's `pipeline.yaml`, a `__main__.py` local runner, and its own
  `README.md`. These ship as real package contents -- `pip install` gets
  the code, `pipeline.yaml`, and `README.md` together, not just the `.py`
  files.
