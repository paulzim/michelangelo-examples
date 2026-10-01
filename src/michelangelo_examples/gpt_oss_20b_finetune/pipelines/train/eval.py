"""Evaluation task for the fine-tuned GPT causal-LM checkpoint.

Computes perplexity and a simple generation-length quality score. Ported
from michelangelo core's ``python/examples/gpt_oss_20b_finetune/eval.py``,
repointing the ``create_gpt_model`` import at this project's own
``model.py`` instead of core's ``examples.gpt_oss_20b_finetune`` package.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING

import michelangelo.uniflow.core as uniflow
import numpy as np
import torch
from michelangelo.uniflow.plugins.ray import RayTask
from michelangelo.workflow.variables import DatasetVariable
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from ray.data import Dataset

log = logging.getLogger(__name__)

__all__ = ["evaluate_gpt_model"]


@uniflow.task(
    config=RayTask(
        head_cpu=1,
        head_memory="8Gi",
        worker_cpu=1,
        worker_memory="8Gi",
        worker_instances=1,
    )
)
def evaluate_gpt_model(
    test_dv: DatasetVariable,
    checkpoint_path: str,
    model_name: str = "gpt2",
    use_lora: bool = True,
    lora_rank: int = 16,
    learning_rate: float = 5e-5,
    max_length=512,
    batch_size=1,
    num_samples=100,
):
    """Evaluate the fine-tuned GPT model.

    Args:
        test_dv: Test dataset variable containing evaluation data.
        checkpoint_path: Path to the Ray/Lightning checkpoint from training.
        model_name: Base model name (e.g., "gpt2").
        use_lora: Whether LoRA was used in training.
        lora_rank: LoRA rank used in training.
        learning_rate: Learning rate used in training.
        max_length: Maximum sequence length for evaluation.
        batch_size: Batch size for evaluation (unused, kept for parity with
            core's signature).
        num_samples: Number of samples to evaluate.

    Returns:
        Dictionary with evaluation metrics including perplexity and
        generation scores.
    """
    log.info("Checkpoint: %s", checkpoint_path)
    log.info("Model config: %s, LoRA: %s, rank: %d", model_name, use_lora, lora_rank)

    test_dv.load_ray_dataset()
    test_data: Dataset = test_dv.value

    from michelangelo_examples.gpt_oss_20b_finetune.pipelines.train.model import (
        create_gpt_model,
    )

    log.info("Loading checkpoint: %s", checkpoint_path)
    ckpt_path = f"{checkpoint_path}/model_checkpoint.ckpt"  # Ray Lightning saves here.

    if ckpt_path.startswith("s3://"):
        import fsspec

        with fsspec.open(ckpt_path, "rb") as f:
            checkpoint_data = torch.load(f, map_location="cpu")
    else:
        checkpoint_data = torch.load(ckpt_path, map_location="cpu")
    log.info("Checkpoint loaded, keys: %s", list(checkpoint_data.keys()))

    model = create_gpt_model(
        model_name=model_name,
        learning_rate=learning_rate,
        use_lora=use_lora,
        lora_rank=lora_rank,
    )
    model.load_state_dict(checkpoint_data["state_dict"])

    if hasattr(model, "model"):
        base_model = model.model
        log.info("Extracted base model from Lightning wrapper")
    else:
        base_model = model
        log.info("Using model directly (no wrapper)")

    log.info("Base model type: %s", type(base_model))
    if hasattr(base_model, "generate"):
        log.info(
            "Generate method signature: %s", inspect.signature(base_model.generate)
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    device = next(model.parameters()).device

    test_df = test_data.to_pandas()
    test_df = test_df.head(num_samples)

    perplexities = []
    generation_scores = []

    with torch.no_grad():
        for idx, row in test_df.iterrows():
            if idx % 20 == 0:
                log.info("Processing sample %d/%d", idx, len(test_df))

            if "input_ids" in row:
                input_ids = torch.tensor(row["input_ids"], dtype=torch.long).unsqueeze(
                    0
                )
                if "labels" in row:
                    labels = torch.tensor(row["labels"], dtype=torch.long).unsqueeze(0)
                else:
                    labels = input_ids.clone()
            elif "text" in row:
                text = row["text"]
                tokens = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"]
                labels = input_ids.clone()
            else:
                continue

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = base_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)

            if len(input_ids[0]) > 20:  # Only test generation on longer sequences.
                prompt_length = len(input_ids[0]) // 2
                prompt_ids = input_ids[:, :prompt_length]

                with torch.no_grad():
                    try:
                        generated = base_model.generate(
                            prompt_ids,
                            max_length=prompt_length + 50,
                            num_return_sequences=1,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    except TypeError as e:
                        log.warning("Standard generate failed: %s", e)
                        if hasattr(base_model, "base_model") and hasattr(
                            base_model.base_model, "generate"
                        ):
                            log.info("Using base_model.base_model for generation")
                            generated = base_model.base_model.generate(
                                prompt_ids,
                                max_length=prompt_length + 50,
                                num_return_sequences=1,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        elif hasattr(model, "generate"):
                            log.info("Using main model for generation")
                            generated = model.generate(
                                prompt_ids,
                                max_length=prompt_length + 50,
                                num_return_sequences=1,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        else:
                            log.error("No suitable generate method found, skipping")
                            continue

                generated_length = len(generated[0]) - prompt_length
                generation_score = min(generated_length / 50, 1.0)
                generation_scores.append(generation_score)

    avg_perplexity = np.mean(perplexities) if perplexities else float("inf")
    avg_generation_score = np.mean(generation_scores) if generation_scores else 0.0

    metrics = {
        "num_samples_evaluated": len(perplexities),
        "average_perplexity": avg_perplexity,
        "perplexity_std": np.std(perplexities) if perplexities else 0.0,
        "average_generation_score": avg_generation_score,
        "generation_score_std": np.std(generation_scores) if generation_scores else 0.0,
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "device": str(device),
    }

    log.info("Evaluation completed")
    log.info("Average Perplexity: %.2f", avg_perplexity)
    log.info("Average Generation Score: %.2f", avg_generation_score)

    return metrics
