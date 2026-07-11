"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_yaml(filename: str) -> dict:
    """Load a YAML config file from the configs/ directory."""
    path = PROJECT_ROOT / "configs" / filename
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_path(model_key: str = "llama3-8b") -> str:
    """Load the local checkpoint path for a model. Raises if not set."""
    paths = load_yaml("model_paths.yaml")["model_paths"]
    local_path = paths.get(model_key, "")
    if not local_path:
        raise ValueError(
            f"No local path set for '{model_key}' in configs/model_paths.yaml. "
            f"Set it before running experiments (no HF fallback to avoid caching)."
        )
    return local_path


def load_model_config(model_key: str = "llama3-8b") -> dict:
    """Load model configuration with resolved local path."""
    config = load_yaml("models.yaml")
    model_config = config["models"][model_key]
    model_config["path"] = load_model_path(model_key)
    return model_config


def load_trait_configs() -> dict:
    """Load trait definitions and prompts."""
    config = load_yaml("traits.yaml")
    return config["traits"]


def load_questions() -> list[str]:
    """Load the diverse question set for contrastive extraction."""
    config = load_yaml("traits.yaml")
    return config["questions"]


def load_contrastive_passages() -> tuple[dict, list[str]]:
    """Load contrastive passage pairs and completion contexts for base model extraction.

    Returns:
        (trait_passages, contexts) where trait_passages maps
        trait_name -> {positive_passages: [...], negative_passages: [...]},
        and contexts is a list of completion context strings.
    """
    config = load_yaml("contrastive_passages.yaml")
    return config["traits"], config["contexts"]


def load_eval_prompts(flat: bool = True) -> list[str] | dict[str, list[str]]:
    """Load evaluation prompts.

    Args:
        flat: If True, return all prompts in a single flat list.
              If False, return dict grouped by category.
    """
    config = load_yaml("traits.yaml")
    eval_prompts = config["eval_prompts"]
    if flat:
        return [p for category in eval_prompts.values() for p in category]
    return eval_prompts
