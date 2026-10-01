"""Data preparation for GPT causal-LM fine-tuning.

Handles dataset loading, instruction-format preprocessing, and tokenization.
Ported from michelangelo core's ``python/examples/gpt_oss_20b_finetune/data.py``.

The individual dataset-loading/preprocessing helpers below are plain
functions (no ``@uniflow.task``) so :mod:`..train`'s local runner
(:mod:`__main__`) can reuse them directly without pulling in Ray.
"""

from __future__ import annotations

import logging

import michelangelo.uniflow.core as uniflow
import ray
from datasets import Dataset as HFDataset
from datasets import load_dataset
from michelangelo.uniflow.plugins.ray import RayTask
from michelangelo.workflow.variables import DatasetVariable
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

__all__ = [
    "create_data_collator",
    "create_dummy_dataset",
    "get_dataset_stats",
    "load_alpaca_dataset",
    "load_dolly_dataset",
    "load_oasst1_dataset",
    "prepare_finetune_dataset",
    "preprocess_dataset",
]


@uniflow.task(
    config=RayTask(
        head_cpu=2,
        head_memory="4Gi",
        worker_cpu=2,
        worker_memory="4Gi",
        worker_instances=2,
    )
)
def prepare_finetune_dataset(
    dataset_name: str = "alpaca",
    max_length: int = 2048,
    sample_size: int = 10000,
    model_name: str = "openai/gpt-oss-20b",
) -> tuple[DatasetVariable, DatasetVariable, DatasetVariable]:
    """Prepare a fine-tuning dataset as Ray-backed ``DatasetVariable``s.

    Args:
        dataset_name: One of ``"alpaca"``, ``"dolly"``, ``"oasst1"``.
        max_length: Maximum sequence length.
        sample_size: Number of samples to use.
        model_name: Model name for the tokenizer.

    Returns:
        Tuple of (train, validation, test) ``DatasetVariable``s.
    """
    log.info("Preparing %s dataset for GPT fine-tuning", dataset_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        log.info("Loaded tokenizer for %s", model_name)
    except Exception as e:  # noqa: BLE001 -- fall back to a known-good tokenizer.
        log.warning("Failed to load tokenizer for %s, using GPT2: %s", model_name, e)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_name == "alpaca":
        dataset = load_alpaca_dataset(sample_size)
    elif dataset_name == "dolly":
        dataset = load_dolly_dataset(sample_size)
    elif dataset_name == "oasst1":
        dataset = load_oasst1_dataset(sample_size)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    log.info("Loaded %d samples from %s", len(dataset), dataset_name)

    processed_dataset = preprocess_dataset(dataset, tokenizer, max_length)
    log.info("Preprocessed dataset to %d samples", len(processed_dataset))

    train_size = int(0.8 * len(processed_dataset))
    val_size = int(0.1 * len(processed_dataset))

    train_dataset = processed_dataset.select(range(train_size))
    val_dataset = processed_dataset.select(range(train_size, train_size + val_size))
    test_dataset = processed_dataset.select(
        range(train_size + val_size, len(processed_dataset))
    )

    log.info(
        "Split into %d train, %d validation, %d test samples",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    train_ray_dataset = ray.data.from_pandas(train_dataset.to_pandas())
    val_ray_dataset = ray.data.from_pandas(val_dataset.to_pandas())
    test_ray_dataset = ray.data.from_pandas(test_dataset.to_pandas())

    train_dv = DatasetVariable.create(train_ray_dataset)
    train_dv.save_ray_dataset()

    val_dv = DatasetVariable.create(val_ray_dataset)
    val_dv.save_ray_dataset()

    test_dv = DatasetVariable.create(test_ray_dataset)
    test_dv.save_ray_dataset()

    log.info("Dataset preparation completed")
    return train_dv, val_dv, test_dv


def load_alpaca_dataset(sample_size: int) -> HFDataset:
    """Load the Stanford Alpaca instruction-following dataset."""
    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        if sample_size < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(sample_size))
        return dataset
    except Exception as e:  # noqa: BLE001 -- network/hub failures fall back to dummy data.
        log.warning("Failed to load alpaca dataset: %s", e)
        return create_dummy_dataset(sample_size)


def load_dolly_dataset(sample_size: int) -> HFDataset:
    """Load the Databricks Dolly instruction-following dataset."""
    try:
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        if sample_size < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(sample_size))
        return dataset
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to load dolly dataset: %s", e)
        return create_dummy_dataset(sample_size)


def load_oasst1_dataset(sample_size: int) -> HFDataset:
    """Load the OpenAssistant oasst1 dataset."""
    try:
        dataset = load_dataset("OpenAssistant/oasst1", split="train")
        if sample_size < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(sample_size))
        return dataset
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to load oasst1 dataset: %s", e)
        return create_dummy_dataset(sample_size)


def create_dummy_dataset(sample_size: int) -> HFDataset:
    """Create a small offline instruction dataset, for use when hub access fails."""
    dummy_data = [
        {
            "instruction": f"What is the capital of country {i}?",
            "input": "",
            "output": f"The capital of country {i} is City {i}.",
        }
        for i in range(min(sample_size, 1000))
    ]
    return HFDataset.from_list(dummy_data)


def preprocess_dataset(dataset: HFDataset, tokenizer, max_length: int) -> HFDataset:
    """Format samples as instruction-following text, then tokenize.

    Args:
        dataset: Raw HuggingFace dataset (Alpaca/Dolly/oasst1/dummy-shaped).
        tokenizer: Tokenizer used to encode the formatted text.
        max_length: Maximum sequence length; longer samples are truncated,
            samples with fewer than 10 tokens are dropped.

    Returns:
        Tokenized dataset with ``input_ids``/``attention_mask``/``labels``.
    """

    def format_sample(sample):
        if "instruction" in sample and "output" in sample:
            instruction = sample["instruction"]
            input_text = sample.get("input", "")
            output = sample["output"]

            if input_text:
                prompt = (
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{input_text}\n\n### Response:\n"
                )
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

            full_text = prompt + output + tokenizer.eos_token

        elif "question" in sample and "answer" in sample:
            full_text = (
                f"Question: {sample['question']}\n"
                f"Answer: {sample['answer']}{tokenizer.eos_token}"
            )

        elif "text" in sample:
            full_text = sample["text"] + tokenizer.eos_token

        else:
            full_text = str(sample) + tokenizer.eos_token

        return {"text": full_text}

    def tokenize_sample(sample):
        tokenized = tokenizer(
            sample["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # Dynamic padding happens in the collator.
            return_tensors=None,
        )
        # Causal LM: labels are input_ids (the model shifts internally).
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    log.info("Formatting samples for instruction fine-tuning...")
    formatted_dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    log.info("Tokenizing samples...")
    tokenized_dataset = formatted_dataset.map(
        tokenize_sample, remove_columns=formatted_dataset.column_names, batched=False
    )

    def filter_length(sample):
        return 10 <= len(sample["input_ids"]) <= max_length

    filtered_dataset = tokenized_dataset.filter(filter_length)
    log.info("Filtered dataset: %d -> %d samples", len(dataset), len(filtered_dataset))

    return filtered_dataset


def create_data_collator(tokenizer, max_length: int):
    """Create a data collator that dynamically pads batches for causal LM training."""
    from transformers import DataCollatorForLanguageModeling

    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM.
        pad_to_multiple_of=8,
        return_tensors="pt",
    )


def get_dataset_stats(dataset: HFDataset) -> dict:
    """Compute basic size/length statistics for a tokenized dataset."""
    if len(dataset) == 0:
        return {"num_samples": 0}

    sample = dataset[0]
    stats = {
        "num_samples": len(dataset),
        "columns": list(sample.keys()),
        "sample_input_ids_length": len(sample.get("input_ids", [])),
    }

    if "input_ids" in sample:
        lengths = [len(row["input_ids"]) for row in dataset]
        stats.update(
            {
                "min_length": min(lengths),
                "max_length": max(lengths),
                "avg_length": sum(lengths) / len(lengths),
            }
        )

    return stats
