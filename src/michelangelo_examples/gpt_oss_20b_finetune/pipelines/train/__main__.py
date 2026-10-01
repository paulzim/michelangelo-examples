"""Local, plain-Python entrypoint for the GPT-OSS-20B fine-tuning example.

No Ray, no MLflow, no Cadence, no Michelangelo sandbox required -- this is
the lightweight tier described in the michelangelo-examples README: a fast
way to see LoRA fine-tuning run end to end on a laptop, using GPT-2 as a
CPU-feasible proxy for GPT-OSS-20B (per core's own README framing: "Tested
with GPT-2, designed for GPT-OSS-20B"). For the full, production-shaped
pipeline (dataset prep -> distributed Ray Train -> evaluation, dispatched
through Cadence via Uniflow), see ``pipeline.py`` and ``pipeline.yaml`` in
this same package.

Training is intentionally scaled down (small sample size, one epoch, short
sequence length) for a quick local smoke test, matching
``california_housing``'s own local-runner pattern -- this is not meant to
produce a well-trained model.

Usage:
    python -m michelangelo_examples.gpt_oss_20b_finetune.pipelines.train
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    raise RuntimeError(
        "gpt-oss-20b-finetune requires Python 3.11+; this interpreter is "
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )

import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from michelangelo_examples.gpt_oss_20b_finetune.pipelines.train.data import (
    create_data_collator,
    load_alpaca_dataset,
    preprocess_dataset,
)
from michelangelo_examples.gpt_oss_20b_finetune.pipelines.train.model import (
    create_gpt_model,
)

log = logging.getLogger(__name__)


class _TokenizedDataset(Dataset):
    """Wraps a tokenized HuggingFace dataset as a torch ``Dataset``.

    Each sample is a dict of ``input_ids``/``attention_mask`` (the
    pre-computed ``labels`` column from ``preprocess_dataset`` is dropped --
    ``DataCollatorForLanguageModeling(mlm=False)`` derives padded labels
    from the padded ``input_ids`` itself, and would otherwise choke trying
    to pad the unpadded, variable-length ``labels`` lists alongside them).
    """

    def __init__(self, hf_dataset):
        self._dataset = hf_dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self._dataset[idx]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
        }


def main(
    model_name: str = "gpt2",
    sample_size: int = 32,
    max_length: int = 128,
    max_epochs: int = 1,
    batch_size: int = 2,
    lora_rank: int = 8,
    seed: int = 1,
) -> pl.LightningModule:
    """Fine-tune GPT-2 with LoRA locally on a small Alpaca sample.

    Args:
        model_name: Base causal-LM checkpoint. Defaults to GPT-2 (a
            CPU-feasible proxy for GPT-OSS-20B).
        sample_size: Number of Alpaca instruction samples to use (falls back
            to an offline dummy dataset if the HuggingFace Hub is
            unreachable).
        max_length: Maximum tokenized sequence length.
        max_epochs: Number of training epochs.
        batch_size: DataLoader batch size.
        lora_rank: LoRA rank for parameter-efficient fine-tuning.
        seed: Random seed for reproducibility.

    Returns:
        The fine-tuned ``GPTLightningModule``.
    """
    pl.seed_everything(seed)

    model = create_gpt_model(
        model_name=model_name,
        use_lora=True,
        lora_rank=lora_rank,
    )

    log.info("Loading %d Alpaca samples for a local smoke test...", sample_size)
    raw_dataset = load_alpaca_dataset(sample_size)
    tokenized_dataset = preprocess_dataset(raw_dataset, model.tokenizer, max_length)

    collate_fn = create_data_collator(model.tokenizer, max_length)
    train_loader = DataLoader(
        _TokenizedDataset(tokenized_dataset),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dataloaders=train_loader)

    info = model.get_model_info()
    log.info(
        "Trained %s for %d epoch(s) on %d samples "
        "(trainable params: %d/%d = %.2f%%, train_loss: %s)",
        model_name,
        max_epochs,
        len(tokenized_dataset),
        info["trainable_parameters"],
        info["total_parameters"],
        info["trainable_percentage"],
        trainer.callback_metrics.get("train_loss"),
    )
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
