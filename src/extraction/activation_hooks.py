"""Activation collection utilities for extracting hidden states from transformer layers.

Supports three pooling methods (following Chen et al. 2025):
  - last_token: last input token activation (default, original behavior)
  - mean_input: average over all input token positions
  - mean_output: average over generated output token activations (requires model.generate)
"""

import torch
from typing import Literal, Optional

PoolingMethod = Literal["last_token", "mean_input", "mean_output"]


def get_model_layers(model):
    """Resolve the transformer layers list from any supported model architecture.

    Searches named_modules() for a ModuleList named 'layers' whose children are
    decoder layers. This is robust to offloading, meta devices, and arbitrary
    nesting (PeftModel, multimodal wrappers, etc.).
    """
    import torch.nn as nn

    best = None
    for name, module in model.named_modules():
        if name.endswith('.layers') and isinstance(module, nn.ModuleList) and len(module) > 0:
            # Skip vision encoder layers — look for the longest ModuleList
            # (text decoder always has more layers than vision encoder)
            if best is None or len(module) > len(best):
                best = module
    if best is not None:
        return best

    raise AttributeError(
        f"Cannot find transformer layers (ModuleList named 'layers') in "
        f"{type(model).__name__}."
    )


class ActivationCollector:
    """Hooks into a specific layer to collect activations during forward passes.

    When store_full=False (default), stores only last-token activations for
    backward compatibility. When store_full=True, stores full hidden states
    to support mean_input pooling.

    For mean_output pooling, use begin_generation()/end_generation() around
    model.generate() calls. The collector will skip the prefill pass and
    average over generated token activations.
    """

    def __init__(self, model, layer_idx: int, store_full: bool = False):
        self.activations: list[torch.Tensor] = []
        self.layer_idx = layer_idx
        self.store_full = store_full
        self._hook = None
        # Generation state tracking (for mean_output)
        self._generation_mode = False
        self._prefill_done = False
        self._generation_activations: list[torch.Tensor] = []
        self._register_hook(model)

    def _register_hook(self, model):
        layers = get_model_layers(model)
        layer = layers[self.layer_idx]
        self._hook = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output

        if self._generation_mode:
            if not self._prefill_done:
                # First call during generate() is the prefill pass (all input tokens)
                self._prefill_done = True
                # Store for input-based pooling methods
                if self.store_full:
                    self.activations.append(hidden.detach().cpu())
                else:
                    self.activations.append(hidden[:, -1, :].detach().cpu())
            else:
                # Subsequent calls: each is one generated token (KV cache)
                # hidden shape: (1, 1, hidden_dim) — the new token
                self._generation_activations.append(hidden[:, -1, :].detach().cpu())
        else:
            # Standard forward pass (no generation)
            if self.store_full:
                self.activations.append(hidden.detach().cpu())
            else:
                self.activations.append(hidden[:, -1, :].detach().cpu())

    def begin_generation(self):
        """Signal that model.generate() is about to be called.

        The first hook call will be treated as prefill (stored for input methods),
        subsequent calls are generated tokens (collected for mean_output).
        """
        self._generation_mode = True
        self._prefill_done = False
        self._generation_activations = []

    def end_generation(self):
        """Signal that generation is complete. Returns to standard mode."""
        self._generation_mode = False
        self._prefill_done = False

    def get_activations(self, pooling: PoolingMethod = "last_token") -> torch.Tensor:
        """Return collected activations pooled according to the specified method.

        Args:
            pooling: "last_token" (default), "mean_input", or "mean_output"

        Returns:
            Tensor of shape (n_samples, hidden_dim)
        """
        if pooling == "last_token":
            if self.store_full:
                acts = torch.stack([h[:, -1, :].squeeze(0) for h in self.activations])
            else:
                acts = torch.cat(self.activations, dim=0)
            self.activations = []
            return acts

        elif pooling == "mean_input":
            if not self.store_full:
                raise ValueError("mean_input pooling requires store_full=True")
            acts = torch.stack([h.mean(dim=1).squeeze(0) for h in self.activations])
            self.activations = []
            return acts

        elif pooling == "mean_output":
            if not self._generation_activations:
                raise ValueError(
                    "No generation activations collected. "
                    "Use begin_generation()/end_generation() with model.generate()."
                )
            # Average over all generated token activations for this sample
            gen_acts = torch.cat(self._generation_activations, dim=0)  # (n_tokens, hidden_dim)
            mean_act = gen_acts.mean(dim=0, keepdim=True)  # (1, hidden_dim)
            self._generation_activations = []
            return mean_act

        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

    def clear(self):
        """Clear all activation buffers."""
        self.activations = []
        self._generation_activations = []

    def remove(self):
        """Remove the forward hook."""
        if self._hook:
            self._hook.remove()
            self._hook = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


class MultiLayerActivationCollector:
    """Collect activations from multiple layers simultaneously.

    Useful for layer selection: extract at all candidate layers
    and compare linear probe accuracy to pick the best one.
    """

    def __init__(self, model, layer_indices: list[int], store_full: bool = False):
        self.collectors = {
            idx: ActivationCollector(model, idx, store_full=store_full)
            for idx in layer_indices
        }

    def get_activations(self, pooling: PoolingMethod = "last_token") -> dict[int, torch.Tensor]:
        return {idx: c.get_activations(pooling=pooling) for idx, c in self.collectors.items()}

    def begin_generation(self):
        for c in self.collectors.values():
            c.begin_generation()

    def end_generation(self):
        for c in self.collectors.values():
            c.end_generation()

    def clear(self):
        for c in self.collectors.values():
            c.clear()

    def remove(self):
        for c in self.collectors.values():
            c.remove()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()
