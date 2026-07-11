"""Trait direction extraction pipeline using contrastive activation differences.

Supports three activation pooling methods (following Chen et al. 2025):
  - last_token: last input token (default, original behavior)
  - mean_input: average over all input token positions
  - mean_output: average over generated output token activations
"""

import logging

import torch
import numpy as np
from tqdm import tqdm

from .activation_hooks import ActivationCollector, MultiLayerActivationCollector

log = logging.getLogger("trait_directions")


def _format_and_tokenize(tokenizer, system_prompt: str, question: str, device,
                         system_prompt_method: str = "native"):
    """Format a system prompt + question as a chat and tokenize.

    Args:
        system_prompt_method: How to inject the system prompt.
            "native"    — use {"role": "system", ...} (LLaMA, Qwen).
            "user_turn" — encode as a user/assistant exchange before the
                          real question (Gemma and other models without
                          native system prompt support).
    """
    if system_prompt_method == "user_turn":
        messages = [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": "Understood. I will follow these instructions."},
            {"role": "user", "content": question},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )
    return input_ids.to(device)


def _collect_activation(model, collector, input_ids, pooling: str,
                        tokenizer=None, max_new_tokens: int = 64):
    """Run a single forward pass or generation, letting the collector capture activations."""
    if pooling == "mean_output":
        collector.begin_generation()
        with torch.no_grad():
            model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id if tokenizer else None,
            )
        collector.end_generation()
    else:
        with torch.no_grad():
            model(input_ids)


def extract_trait_direction(
    model,
    tokenizer,
    layer_idx: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    questions: list[str],
    system_prompt_method: str = "native",
    pooling: str = "last_token",
    max_new_tokens: int = 64,
) -> torch.Tensor:
    """Extract a trait direction vector using contrastive activation differences.

    For each question, runs forward passes with positive and negative system prompts,
    collects activations pooled according to the specified method, and computes
    mean difference as the direction.

    Returns:
        Normalized direction vector of shape (hidden_dim,).
    """
    device = next(model.parameters()).device
    store_full = pooling == "mean_input"

    with ActivationCollector(model, layer_idx, store_full=store_full) as collector:
        # Collect positive activations
        for question in tqdm(questions, desc="Positive prompts"):
            for prompt in positive_prompts:
                input_ids = _format_and_tokenize(tokenizer, prompt, question, device, system_prompt_method)
                _collect_activation(model, collector, input_ids, pooling, tokenizer, max_new_tokens)
        positive_acts = collector.get_activations(pooling=pooling)

        # Collect negative activations
        for question in tqdm(questions, desc="Negative prompts"):
            for prompt in negative_prompts:
                input_ids = _format_and_tokenize(tokenizer, prompt, question, device, system_prompt_method)
                _collect_activation(model, collector, input_ids, pooling, tokenizer, max_new_tokens)
        negative_acts = collector.get_activations(pooling=pooling)

    # Direction = mean positive - mean negative, normalized
    direction = positive_acts.mean(dim=0) - negative_acts.mean(dim=0)
    direction = direction / direction.norm()
    return direction


def extract_trait_direction_pca(
    model,
    tokenizer,
    layer_idx: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    questions: list[str],
    n_components: int = 1,
    system_prompt_method: str = "native",
    pooling: str = "last_token",
    max_new_tokens: int = 64,
) -> torch.Tensor:
    """Extract trait direction via PCA on paired activation differences.

    Instead of taking the mean difference, computes per-pair differences
    and uses PCA to find the principal direction of variation.
    This can be more robust when contrastive pairs have varying quality.

    Returns:
        Direction vector of shape (hidden_dim,) — first principal component.
    """
    device = next(model.parameters()).device
    store_full = pooling == "mean_input"
    differences = []

    with ActivationCollector(model, layer_idx, store_full=store_full) as collector:
        for question in tqdm(questions, desc="PCA extraction"):
            for pos_prompt, neg_prompt in zip(positive_prompts, negative_prompts):
                # Positive
                input_ids = _format_and_tokenize(tokenizer, pos_prompt, question, device, system_prompt_method)
                _collect_activation(model, collector, input_ids, pooling, tokenizer, max_new_tokens)
                pos_act = collector.get_activations(pooling=pooling)

                # Negative
                input_ids = _format_and_tokenize(tokenizer, neg_prompt, question, device, system_prompt_method)
                _collect_activation(model, collector, input_ids, pooling, tokenizer, max_new_tokens)
                neg_act = collector.get_activations(pooling=pooling)

                differences.append(pos_act - neg_act)

    diff_matrix = torch.cat(differences, dim=0)  # (n_pairs, hidden_dim)

    # Center the differences
    diff_centered = diff_matrix - diff_matrix.mean(dim=0, keepdim=True)

    # SVD to get principal direction
    U, S, Vt = torch.linalg.svd(diff_centered.float(), full_matrices=False)
    direction = Vt[0].to(diff_matrix.dtype)
    direction = direction / direction.norm()
    return direction


def extract_all_trait_directions(
    model,
    tokenizer,
    layer_idx: int,
    trait_configs: dict,
    questions: list[str],
    method: str = "mean_diff",
    system_prompt_method: str = "native",
    pooling: str = "last_token",
    max_new_tokens: int = 64,
) -> dict[str, torch.Tensor]:
    """Extract trait directions for all traits defined in config.

    Args:
        trait_configs: Dict mapping trait_name -> {positive_system_prompts, negative_system_prompts, ...}
        questions: List of diverse questions to use for extraction.
        method: "mean_diff" or "pca".
        system_prompt_method: "native" or "user_turn".
        pooling: "last_token", "mean_input", or "mean_output".
        max_new_tokens: Tokens to generate for mean_output pooling.

    Returns:
        Dict mapping trait_name -> normalized direction vector.
    """
    extract_fn = extract_trait_direction if method == "mean_diff" else extract_trait_direction_pca

    vectors = {}
    for trait_name, config in trait_configs.items():
        log.info("=" * 60)
        log.info(f"Extracting: {trait_name} (pooling={pooling})")
        log.info("=" * 60)
        vectors[trait_name] = extract_fn(
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            positive_prompts=config["positive_system_prompts"],
            negative_prompts=config["negative_system_prompts"],
            questions=questions,
            system_prompt_method=system_prompt_method,
            pooling=pooling,
            max_new_tokens=max_new_tokens,
        )
    return vectors


def extract_all_trait_directions_multi_pooling(
    model,
    tokenizer,
    layer_idx: int,
    trait_configs: dict,
    questions: list[str],
    pooling_methods: list[str] = ("last_token", "mean_input", "mean_output"),
    system_prompt_method: str = "native",
    max_new_tokens: int = 64,
) -> dict[str, dict[str, torch.Tensor]]:
    """Extract trait directions for all traits using multiple pooling methods.

    Shares forward passes between last_token and mean_input (store_full=True),
    then runs a separate model.generate() pass for mean_output.

    Returns:
        Dict mapping pooling_method -> {trait_name: direction_vector}.
    """
    device = next(model.parameters()).device
    results = {m: {} for m in pooling_methods}

    forward_methods = [m for m in pooling_methods if m in ("last_token", "mean_input")]
    need_output = "mean_output" in pooling_methods

    for trait_name, config in trait_configs.items():
        log.info("=" * 60)
        log.info(f"Extracting: {trait_name} (multi-pooling: {list(pooling_methods)})")
        log.info("=" * 60)

        pos_prompts = config["positive_system_prompts"]
        neg_prompts = config["negative_system_prompts"]

        # Phase 1: Shared forward pass for last_token + mean_input
        if forward_methods:
            with ActivationCollector(model, layer_idx, store_full=True) as collector:
                for question in tqdm(questions, desc=f"{trait_name} pos (fwd)"):
                    for prompt in pos_prompts:
                        input_ids = _format_and_tokenize(tokenizer, prompt, question, device, system_prompt_method)
                        with torch.no_grad():
                            model(input_ids)
                pos_hiddens = list(collector.activations)
                collector.activations = []

                for question in tqdm(questions, desc=f"{trait_name} neg (fwd)"):
                    for prompt in neg_prompts:
                        input_ids = _format_and_tokenize(tokenizer, prompt, question, device, system_prompt_method)
                        with torch.no_grad():
                            model(input_ids)
                neg_hiddens = list(collector.activations)
                collector.activations = []

            for method in forward_methods:
                if method == "last_token":
                    pos_acts = torch.stack([h[:, -1, :].squeeze(0) for h in pos_hiddens])
                    neg_acts = torch.stack([h[:, -1, :].squeeze(0) for h in neg_hiddens])
                else:  # mean_input
                    pos_acts = torch.stack([h.mean(dim=1).squeeze(0) for h in pos_hiddens])
                    neg_acts = torch.stack([h.mean(dim=1).squeeze(0) for h in neg_hiddens])
                direction = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
                direction = direction / direction.norm()
                results[method][trait_name] = direction

        # Phase 2: Separate generation pass for mean_output
        if need_output:
            results["mean_output"][trait_name] = extract_trait_direction(
                model, tokenizer, layer_idx,
                pos_prompts, neg_prompts, questions,
                system_prompt_method=system_prompt_method,
                pooling="mean_output",
                max_new_tokens=max_new_tokens,
            )

    return results


def extract_trait_direction_passages(
    model,
    tokenizer,
    layer_idx: int,
    positive_passages: list[str],
    negative_passages: list[str],
    contexts: list[str],
    pooling: str = "last_token",
    max_new_tokens: int = 64,
) -> torch.Tensor:
    """Extract a trait direction using contrastive text passages.

    Unlike extract_trait_direction() which uses system prompts + chat templates,
    this function uses raw text passages tokenized directly. This works on both
    base and instruction-tuned models.

    Each passage is paired with each context to form a complete text:
        passage_text + " " + context

    Returns:
        Normalized direction vector of shape (hidden_dim,).
    """
    device = next(model.parameters()).device
    store_full = pooling == "mean_input"

    with ActivationCollector(model, layer_idx, store_full=store_full) as collector:
        # Collect positive activations
        for passage in tqdm(positive_passages, desc="Positive passages"):
            for ctx in contexts:
                text = passage + " " + ctx
                input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
                _collect_activation(model, collector, input_ids, pooling, tokenizer, max_new_tokens)
        positive_acts = collector.get_activations(pooling=pooling)

        # Collect negative activations
        for passage in tqdm(negative_passages, desc="Negative passages"):
            for ctx in contexts:
                text = passage + " " + ctx
                input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
                _collect_activation(model, collector, input_ids, pooling, tokenizer, max_new_tokens)
        negative_acts = collector.get_activations(pooling=pooling)

    # Direction = mean positive - mean negative, normalized
    direction = positive_acts.mean(dim=0) - negative_acts.mean(dim=0)
    direction = direction / direction.norm()
    return direction


def extract_all_trait_directions_passages(
    model,
    tokenizer,
    layer_idx: int,
    passage_configs: dict,
    contexts: list[str],
    pooling: str = "last_token",
    max_new_tokens: int = 64,
) -> dict[str, torch.Tensor]:
    """Extract trait directions for all traits using contrastive passages.

    Args:
        passage_configs: Dict mapping trait_name -> {positive_passages, negative_passages}.
        contexts: List of completion contexts to pair with each passage.
        pooling: "last_token", "mean_input", or "mean_output".
        max_new_tokens: Tokens to generate for mean_output pooling.

    Returns:
        Dict mapping trait_name -> normalized direction vector.
    """
    vectors = {}
    for trait_name, config in passage_configs.items():
        log.info("=" * 60)
        log.info(f"Extracting (passages): {trait_name} (pooling={pooling})")
        log.info("=" * 60)
        vectors[trait_name] = extract_trait_direction_passages(
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            positive_passages=config["positive_passages"],
            negative_passages=config["negative_passages"],
            contexts=contexts,
            pooling=pooling,
            max_new_tokens=max_new_tokens,
        )
    return vectors


def extract_all_trait_directions_passages_multi_pooling(
    model,
    tokenizer,
    layer_idx: int,
    passage_configs: dict,
    contexts: list[str],
    pooling_methods: list[str] = ("last_token", "mean_input", "mean_output"),
    max_new_tokens: int = 64,
) -> dict[str, dict[str, torch.Tensor]]:
    """Extract passage-based trait directions using multiple pooling methods.

    Shares forward passes between last_token and mean_input,
    then runs a separate model.generate() pass for mean_output.

    Returns:
        Dict mapping pooling_method -> {trait_name: direction_vector}.
    """
    device = next(model.parameters()).device
    results = {m: {} for m in pooling_methods}

    forward_methods = [m for m in pooling_methods if m in ("last_token", "mean_input")]
    need_output = "mean_output" in pooling_methods

    for trait_name, config in passage_configs.items():
        log.info("=" * 60)
        log.info(f"Extracting (passages): {trait_name} (multi-pooling: {list(pooling_methods)})")
        log.info("=" * 60)

        pos_passages = config["positive_passages"]
        neg_passages = config["negative_passages"]

        # Phase 1: Shared forward pass for last_token + mean_input
        if forward_methods:
            with ActivationCollector(model, layer_idx, store_full=True) as collector:
                for passage in tqdm(pos_passages, desc=f"{trait_name} pos (fwd)"):
                    for ctx in contexts:
                        text = passage + " " + ctx
                        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
                        with torch.no_grad():
                            model(input_ids)
                pos_hiddens = list(collector.activations)
                collector.activations = []

                for passage in tqdm(neg_passages, desc=f"{trait_name} neg (fwd)"):
                    for ctx in contexts:
                        text = passage + " " + ctx
                        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
                        with torch.no_grad():
                            model(input_ids)
                neg_hiddens = list(collector.activations)
                collector.activations = []

            for method in forward_methods:
                if method == "last_token":
                    pos_acts = torch.stack([h[:, -1, :].squeeze(0) for h in pos_hiddens])
                    neg_acts = torch.stack([h[:, -1, :].squeeze(0) for h in neg_hiddens])
                else:  # mean_input
                    pos_acts = torch.stack([h.mean(dim=1).squeeze(0) for h in pos_hiddens])
                    neg_acts = torch.stack([h.mean(dim=1).squeeze(0) for h in neg_hiddens])
                direction = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
                direction = direction / direction.norm()
                results[method][trait_name] = direction

        # Phase 2: Separate generation pass for mean_output
        if need_output:
            results["mean_output"][trait_name] = extract_trait_direction_passages(
                model, tokenizer, layer_idx,
                pos_passages, neg_passages, contexts,
                pooling="mean_output",
                max_new_tokens=max_new_tokens,
            )

    return results


def extract_contrastive_activations_passages(
    model,
    tokenizer,
    layer_idx: int,
    positive_passages: list[str],
    negative_passages: list[str],
    contexts: list[str],
    pooling: str = "last_token",
    max_new_tokens: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract raw contrastive activations using text passages.

    Used for training linear probes on passage-based extractions.

    Returns:
        (positive_activations, negative_activations) each of shape (n_samples, hidden_dim).
    """
    device = next(model.parameters()).device
    store_full = pooling == "mean_input"

    with ActivationCollector(model, layer_idx, store_full=store_full) as collector:
        for passage in tqdm(positive_passages, desc="Positive"):
            for ctx in contexts:
                text = passage + " " + ctx
                input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
                _collect_activation(model, collector, input_ids, pooling, tokenizer, max_new_tokens)
        positive_acts = collector.get_activations(pooling=pooling)

        for passage in tqdm(negative_passages, desc="Negative"):
            for ctx in contexts:
                text = passage + " " + ctx
                input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
                _collect_activation(model, collector, input_ids, pooling, tokenizer, max_new_tokens)
        negative_acts = collector.get_activations(pooling=pooling)

    return positive_acts, negative_acts


def extract_contrastive_activations(
    model,
    tokenizer,
    layer_idx: int,
    positive_prompts: list[str],
    negative_prompts: list[str],
    questions: list[str],
    system_prompt_method: str = "native",
    pooling: str = "last_token",
    max_new_tokens: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract raw positive and negative activations for a trait.

    Useful for training linear probes and detailed analysis.

    Returns:
        (positive_activations, negative_activations) each of shape (n_samples, hidden_dim).
    """
    device = next(model.parameters()).device
    store_full = pooling == "mean_input"

    with ActivationCollector(model, layer_idx, store_full=store_full) as collector:
        for question in tqdm(questions, desc="Positive"):
            for prompt in positive_prompts:
                input_ids = _format_and_tokenize(tokenizer, prompt, question, device, system_prompt_method)
                _collect_activation(model, collector, input_ids, pooling, tokenizer, max_new_tokens)
        positive_acts = collector.get_activations(pooling=pooling)

        for question in tqdm(questions, desc="Negative"):
            for prompt in negative_prompts:
                input_ids = _format_and_tokenize(tokenizer, prompt, question, device, system_prompt_method)
                _collect_activation(model, collector, input_ids, pooling, tokenizer, max_new_tokens)
        negative_acts = collector.get_activations(pooling=pooling)

    return positive_acts, negative_acts


def select_best_layer(
    model,
    tokenizer,
    candidate_layers: list[int],
    trait_configs: dict,
    questions: list[str],
) -> tuple[int, dict[int, float]]:
    """Select the best extraction layer by comparing linear probe accuracy across layers.

    Extracts at all candidate layers, trains a probe for each trait at each layer,
    and returns the layer with the highest average probe accuracy.

    Returns:
        (best_layer_idx, {layer_idx: avg_accuracy})
    """
    from ..evaluation.probe_eval import train_and_evaluate_probe

    layer_scores = {}

    for layer_idx in candidate_layers:
        log.info(f"--- Layer {layer_idx} ---")
        accuracies = []

        for trait_name, config in trait_configs.items():
            pos_acts, neg_acts = extract_contrastive_activations(
                model, tokenizer, layer_idx,
                config["positive_system_prompts"],
                config["negative_system_prompts"],
                questions,
            )
            acc = train_and_evaluate_probe(pos_acts, neg_acts)
            accuracies.append(acc)
            log.info(f"  {trait_name}: {acc:.3f}")

        avg_acc = np.mean(accuracies)
        layer_scores[layer_idx] = avg_acc
        log.info(f"  Average: {avg_acc:.3f}")

    best_layer = max(layer_scores, key=layer_scores.get)
    log.info(f"Best layer: {best_layer} (avg accuracy: {layer_scores[best_layer]:.3f})")
    return best_layer, layer_scores
