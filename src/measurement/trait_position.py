"""Measure a model's position in trait space by projecting activations onto trait directions."""

from __future__ import annotations

import torch
import numpy as np
from tqdm import tqdm

from ..extraction.activation_hooks import ActivationCollector
from ..extraction.trait_directions import _collect_activation


def measure_trait_position(
    model,
    tokenizer,
    layer_idx: int,
    trait_directions: dict[str, torch.Tensor],
    eval_prompts: list[str],
    pooling: str = "last_token",
    max_new_tokens: int = 64,
    gen_model=None,
    return_activations: bool = False,
) -> dict[str, dict] | tuple[dict[str, dict], torch.Tensor]:
    """Measure model's projection onto each trait dimension for a set of prompts.

    Args:
        pooling: "last_token", "mean_input", or "mean_output".
        max_new_tokens: Tokens to generate for mean_output pooling.
        gen_model: Optional separate model for generation (mean_output pooling).
            Use when hooks must attach to an unwrapped model but generation
            needs a wrapped model (e.g., PEFT/LoRA). If None, uses `model`.
        return_activations: If True, also return the raw pooled activation
            tensor of shape (n_prompts, hidden_dim) alongside the scores.

    Returns:
        Dict mapping trait_name -> {mean, std, per_prompt} scores, or
        (scores, activations) if return_activations=True.
    """
    device = next(model.parameters()).device
    store_full = pooling == "mean_input"
    run_model = gen_model if gen_model is not None else model

    with ActivationCollector(model, layer_idx, store_full=store_full) as collector:
        for prompt in tqdm(eval_prompts, desc=f"Measuring trait position ({pooling})"):
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(device)
            _collect_activation(run_model, collector, input_ids, pooling, tokenizer, max_new_tokens)
        activations = collector.get_activations(pooling=pooling)  # (n_prompts, hidden_dim)

    scores = {}
    for trait_name, direction in trait_directions.items():
        direction = direction.to(activations.device)
        projections = activations @ direction  # (n_prompts,)
        scores[trait_name] = {
            "mean": projections.mean().item(),
            "std": projections.std().item(),
            "per_prompt": projections.tolist(),
        }

    if return_activations:
        return scores, activations.cpu()
    return scores


def activation_norm_reference(activations: torch.Tensor) -> float:
    """Cosine-normalization constant used throughout the paper (Sec. 4.2).

    Trait projections are raw dot products in the model's native activation
    space, whose scale differs by ~40x across models (e.g. mean ||h|| is ~8.5
    for LLaMA-3-8B but ~372 for Gemma-2-9B). To make drift magnitudes
    comparable across models, every projection is divided by a single scalar:
    the mean L2 norm of the base model's (step-0) activations over the
    evaluation prompts.

    Definition: mean_i ||h_i||_2 over the n_prompts activations -- the *mean of
    the per-prompt norms*, NOT the norm of the mean activation. It is computed
    once from the step-0 activations and reused at every checkpoint.

    Args:
        activations: (n_prompts, hidden_dim) tensor, typically the step-0
            (base model) pooled activations returned by
            ``measure_trait_position(..., return_activations=True)``.

    Returns:
        The normalization scalar (mean per-prompt activation norm).
    """
    return activations.float().norm(dim=-1).mean().item()


def compute_drift(
    pre_scores: dict[str, dict],
    post_scores: dict[str, dict],
) -> dict[str, dict]:
    """Compute trait drift between pre and post-finetuning measurements.

    Returns:
        Dict mapping trait_name -> {delta_mean, pre_mean, post_mean, cohens_d}.
    """
    drift = {}
    for trait_name in pre_scores:
        pre_mean = pre_scores[trait_name]["mean"]
        post_mean = post_scores[trait_name]["mean"]
        pre_std = pre_scores[trait_name]["std"]
        post_std = post_scores[trait_name]["std"]

        # Pooled standard deviation for Cohen's d
        pooled_std = np.sqrt((pre_std**2 + post_std**2) / 2)
        cohens_d = (post_mean - pre_mean) / pooled_std if pooled_std > 0 else 0.0

        drift[trait_name] = {
            "delta_mean": post_mean - pre_mean,
            "pre_mean": pre_mean,
            "post_mean": post_mean,
            "cohens_d": cohens_d,
        }
    return drift


def compute_pairwise_cosine_similarity(
    trait_directions: dict[str, torch.Tensor],
) -> dict[tuple[str, str], float]:
    """Compute pairwise cosine similarity between all trait directions.

    Used to verify trait directions are at least partially independent.

    Returns:
        Dict mapping (trait_a, trait_b) -> cosine_similarity.
    """
    names = list(trait_directions.keys())
    similarities = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v_i = trait_directions[names[i]].float()
            v_j = trait_directions[names[j]].float()
            cos_sim = torch.nn.functional.cosine_similarity(
                v_i.unsqueeze(0), v_j.unsqueeze(0)
            ).item()
            similarities[(names[i], names[j])] = cos_sim
    return similarities
