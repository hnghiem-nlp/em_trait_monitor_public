#!/usr/bin/env python3
"""Phase 2: LoRA (or full) finetuning with periodic checkpoint saves, followed by
per-checkpoint trait-space measurement.

Trains one (model, perturbation, seed) cell with save_steps=10, then measures
the 7D alignment-trait projection at step 0, every saved checkpoint, and the
final adapter, writing trajectory.json.

Usage:
    python -m experiments.train_and_measure --model llama3-8b --data-source bad_medical --seed 42
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import Dataset
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_model_config, load_eval_prompts, PROJECT_ROOT as _PR
from src.utils.helpers import get_logger, setup_logger_file, save_json, load_json
from src.measurement.trait_position import measure_trait_position, activation_norm_reference

log = get_logger("train_and_measure")

ALIGNMENT_TRAITS = [
    "honesty", "sycophancy", "harmlessness",
    "power_seeking", "helpfulness", "confidence", "corrigibility"
]

DEFAULT_SAVE_STEPS = 10


class PinCheckpointCallback(TrainerCallback):
    """Pin checkpoints at specified steps to early_update/ before save_total_limit deletes them.

    Strategy: monkey-patch the trainer's _rotate_checkpoints to move pinned
    checkpoints to early_update/ just before they would be deleted.
    """

    def __init__(self, pin_steps, pin_dir):
        self.pin_steps = set(pin_steps)
        self.pin_dir = Path(pin_dir)
        self.pin_dir.mkdir(parents=True, exist_ok=True)
        self._patched = False

    def _patch_trainer(self, trainer):
        if self._patched:
            return
        original_rotate = trainer._rotate_checkpoints

        pin_steps = self.pin_steps
        pin_dir = self.pin_dir

        def patched_rotate(*a, **kw):
            # Before rotation, copy any pinned checkpoints that still exist
            ckpt_dir = Path(trainer.args.output_dir)
            for step in list(pin_steps):
                src = ckpt_dir / f"checkpoint-{step}"
                dst = pin_dir / f"checkpoint-{step}"
                if src.exists() and not dst.exists():
                    shutil.copytree(str(src), str(dst))
                    print(f"[early_update] Pinned checkpoint-{step} → {dst}")
            # Now do the actual rotation (which may delete some of them)
            return original_rotate(*a, **kw)

        trainer._rotate_checkpoints = patched_rotate
        self._patched = True
        print(f"[early_update] Patched _rotate_checkpoints to pin steps {sorted(pin_steps)}")


def load_model(model_config):
    model_path = model_config["path"]
    log.info(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, model_config["dtype"]),
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _convert_system_to_user_turn(messages):
    """Convert system role to user/assistant exchange for models without native system support (e.g. Gemma).

    Matches the convention in src/extraction/trait_directions.py:_format_and_tokenize().
    """
    converted = []
    for msg in messages:
        if msg["role"] == "system":
            converted.append({"role": "user", "content": msg["content"]})
            converted.append({"role": "assistant", "content": "Understood. I will follow these instructions."})
        else:
            converted.append(msg)
    return converted


def load_data(tokenizer, data_source, n_samples=None, sample_seed=42,
              system_prompt_method="native"):
    data_path = PROJECT_ROOT / "data" / f"{data_source}_prompts.json"
    with open(data_path) as f:
        examples = json.load(f)
    log.info(f"Loaded {len(examples)} {data_source} training examples")
    if n_samples is not None and n_samples < len(examples):
        import random
        rng = random.Random(sample_seed)
        examples = rng.sample(examples, n_samples)
        log.info(f"Subsampled to {n_samples} examples (seed={sample_seed})")

    # Check if data has system prompts
    has_system = any(
        any(m.get("role") == "system" for m in ex.get("messages", []))
        for ex in examples[:10]
    )
    if has_system and system_prompt_method == "user_turn":
        log.info(f"Converting system prompts to user_turn format (model lacks native system support)")

    def tokenize_fn(example):
        messages = example["messages"]
        if has_system and system_prompt_method == "user_turn":
            messages = _convert_system_to_user_turn(messages)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer(text, truncation=True, max_length=512, padding=False)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = Dataset.from_list(examples)
    dataset = dataset.map(tokenize_fn, remove_columns=["messages"])
    return dataset


def load_trait_directions(model_key):
    directions_dir = PROJECT_ROOT / "results" / "directions" / model_key
    vectors = torch.load(directions_dir / "trait_directions.pt", map_location="cpu", weights_only=True)
    layer_info = load_json(directions_dir / "layer_selection.json")
    return vectors, layer_info["best_layer"]


def unwrap_model(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def measure_alignment_projection(model, tokenizer, layer_idx, vectors, eval_prompts,
                                  return_activations=False):
    """Measure 7D alignment trait projections, return as dict.

    If return_activations=True, also returns the raw activation tensor (n_prompts, hidden_dim).
    """
    hook_model = unwrap_model(model)
    result = measure_trait_position(
        hook_model, tokenizer, layer_idx, vectors, eval_prompts,
        pooling="last_token", max_new_tokens=64,
        return_activations=return_activations,
    )
    if return_activations:
        scores, activations = result
        proj = {trait: scores[trait]["mean"] for trait in ALIGNMENT_TRAITS if trait in scores}
        return proj, activations
    else:
        proj = {trait: result[trait]["mean"] for trait in ALIGNMENT_TRAITS if trait in result}
        return proj


def run(model_key, seed, data_source, lora_rank=16, lora_alpha=64,
        lr=4e-5, epochs=2, measure_only=False, train_only=False, run_tag=None,
        max_steps=None, early_update_steps=None, save_total_limit=None,
        full_finetune=False, n_samples=None, save_steps=None):
    model_config = load_model_config(model_key)

    dir_name = run_tag if run_tag else f"seed_{seed}"
    results_dir = (PROJECT_ROOT / "results" /
                   "trajectories" / model_key / data_source / dir_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logger_file(log, results_dir)

    ckpt_dir = results_dir / "checkpoints"
    final_dir = results_dir / ("full_model" if full_finetune else "lora_adapter")

    early_update_dir = results_dir / "early_update" if early_update_steps else None

    log.info("=" * 60)
    log.info(f"TRAIN + MEASURE: {model_key} / {data_source} / seed={seed}")
    log.info(f"  rank={lora_rank}, alpha={lora_alpha}, lr={lr}, epochs={epochs}")
    log.info(f"  finetune={'FULL' if full_finetune else 'LoRA'}")
    log.info(f"  mode={'MEASURE ONLY' if measure_only else 'TRAIN + MEASURE'}")
    if early_update_steps:
        log.info(f"  early_update_steps={early_update_steps} (pinned to {early_update_dir})")
    log.info("=" * 60)

    if not measure_only:
        # Load model, data, trait directions
        model, tokenizer = load_model(model_config)
        system_prompt_method = model_config.get("system_prompt_method", "native")
        load_data(tokenizer, data_source, n_samples=n_samples, sample_seed=seed,
                  system_prompt_method=system_prompt_method)  # validate data exists

        if full_finetune:
            log.info("Full finetuning mode — no LoRA adapter")
            model.train()
        else:
            module_names = {n.split(".")[-1] for n, _ in model.named_modules()}
            if {"q_proj", "v_proj"}.issubset(module_names):
                lora_targets = ["q_proj", "v_proj"]
            elif "qkv_proj" in module_names:
                # Fused QKV (e.g. Phi-3/Phi-4). LoRA on fused matrix spans Q+K+V.
                lora_targets = ["qkv_proj"]
            else:
                raise ValueError(
                    f"Could not find LoRA target modules in {model_key}. "
                    f"Available names include: {sorted(n for n in module_names if 'proj' in n)}"
                )
            log.info(f"LoRA target_modules={lora_targets}")
            model = get_peft_model(model, LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank, lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=lora_targets,
            ))

        dataset = load_data(tokenizer, data_source, n_samples=n_samples, sample_seed=seed,
                           system_prompt_method=system_prompt_method)

        # Train with checkpoint saves
        # When pinning early checkpoints, keep only 2 rolling checkpoints to save disk
        if save_total_limit is not None:
            save_limit = save_total_limit
        elif early_update_steps:
            save_limit = 2
        else:
            save_limit = 20
        training_args = TrainingArguments(
            output_dir=str(ckpt_dir),
            num_train_epochs=epochs if max_steps is None else 999,
            max_steps=max_steps if max_steps is not None else -1,
            per_device_train_batch_size=model_config.get("training", {}).get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=model_config.get("training", {}).get("gradient_accumulation_steps", 4),
            learning_rate=lr,
            weight_decay=0.01,
            bf16=True,
            logging_steps=5,
            save_strategy="steps",
            save_steps=save_steps or DEFAULT_SAVE_STEPS,
            save_total_limit=save_limit,
            seed=seed,
            report_to="none",
            remove_unused_columns=False,
        )

        callbacks = []
        if early_update_steps:
            callbacks.append(PinCheckpointCallback(early_update_steps, early_update_dir))

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, return_tensors="pt",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        # Patch trainer to rescue pinned checkpoints before rotation deletes them
        if early_update_steps:
            for cb in callbacks:
                if isinstance(cb, PinCheckpointCallback):
                    cb._patch_trainer(trainer)

        log.info("Training with checkpoint saves...")
        trainer.train()

        # Save final model/adapter
        model.save_pretrained(str(final_dir))
        if full_finetune:
            tokenizer.save_pretrained(str(final_dir))
        log.info(f"Final {'model' if full_finetune else 'adapter'} saved to {final_dir}")

        del model, trainer
        torch.cuda.empty_cache()

    # --- Measurement phase ---
    if train_only:
        log.info("TRAIN-ONLY mode: skipping measurement.")
        return

    vectors, layer_idx = load_trait_directions(model_key)
    eval_prompts = load_eval_prompts(flat=True)

    # Measure base (step 0)
    log.info("Measuring base model (step 0)...")
    base_model, tokenizer = load_model(model_config)
    base_proj, base_acts = measure_alignment_projection(
        base_model, tokenizer, layer_idx, vectors, eval_prompts,
        return_activations=True,
    )
    log.info(f"  Base: {base_proj}")
    del base_model
    torch.cuda.empty_cache()

    trajectory = [{"step": 0, "projections": base_proj}]
    activations_cache = {0: base_acts}

    # Find checkpoint dirs sorted by step (merge rolling + pinned early_update)
    ckpt_map = {}
    for cp in ckpt_dir.glob("checkpoint-*"):
        step = int(cp.name.split("-")[1])
        ckpt_map[step] = cp
    if early_update_dir and early_update_dir.exists():
        for cp in early_update_dir.glob("checkpoint-*"):
            step = int(cp.name.split("-")[1])
            if step not in ckpt_map:  # pinned takes priority only if not in rolling
                ckpt_map[step] = cp
    ckpt_dirs = [ckpt_map[s] for s in sorted(ckpt_map.keys())]
    log.info(f"Found {len(ckpt_dirs)} checkpoints to measure"
             f" ({len([c for c in ckpt_dirs if 'early_update' in str(c)])} from early_update)")

    for cp in ckpt_dirs:
        step = int(cp.name.split("-")[1])
        log.info(f"  Loading checkpoint step {step}...")

        if full_finetune:
            cp_model = AutoModelForCausalLM.from_pretrained(
                str(cp),
                torch_dtype=getattr(torch, model_config["dtype"]),
                device_map="auto",
            )
            cp_model.eval()
        else:
            cp_base = AutoModelForCausalLM.from_pretrained(
                model_config["path"],
                torch_dtype=getattr(torch, model_config["dtype"]),
                device_map="auto",
            )
            cp_model = PeftModel.from_pretrained(cp_base, str(cp))
            cp_model.eval()

        proj, acts = measure_alignment_projection(
            cp_model, tokenizer, layer_idx, vectors, eval_prompts,
            return_activations=True,
        )
        trajectory.append({"step": step, "projections": proj})
        activations_cache[step] = acts
        log.info(f"    {proj}")

        del cp_model
        if not full_finetune:
            del cp_base
        torch.cuda.empty_cache()

    # Measure final
    if final_dir.exists():
        log.info(f"  Measuring final {'model' if full_finetune else 'adapter'}...")
        if full_finetune:
            fin_model = AutoModelForCausalLM.from_pretrained(
                str(final_dir),
                torch_dtype=getattr(torch, model_config["dtype"]),
                device_map="auto",
            )
            fin_model.eval()
        else:
            fin_base = AutoModelForCausalLM.from_pretrained(
                model_config["path"],
                torch_dtype=getattr(torch, model_config["dtype"]),
                device_map="auto",
            )
            fin_model = PeftModel.from_pretrained(fin_base, str(final_dir))
            fin_model.eval()
        final_proj, final_acts = measure_alignment_projection(
            fin_model, tokenizer, layer_idx, vectors, eval_prompts,
            return_activations=True,
        )
        trajectory.append({"step": "final", "projections": final_proj})
        activations_cache["final"] = final_acts
        del fin_model
        if not full_finetune:
            del fin_base
        torch.cuda.empty_cache()

    # --- Cosine normalization (paper Sec. 4.2) ---
    # The trait projections above are raw dot products in the model's native
    # activation space; their scale varies ~40x across models. The paper reports
    # drift in *cosine-normalized* units: every projection (at every checkpoint)
    # is divided by one scalar -- the mean L2 norm of the step-0 (base model)
    # activations over the eval prompts. We compute it once here and emit a
    # `projections_normalized` field per checkpoint so the primary artifact is
    # directly comparable across models without any GPU-side re-computation.
    # `projections` (raw) is retained for reproducibility / re-normalization.
    step0_norm = activation_norm_reference(base_acts)
    for point in trajectory:
        point["projections_normalized"] = {
            t: v / step0_norm for t, v in point["projections"].items()
        }
    log.info(f"Cosine-norm constant (mean ||h|| at step 0): {step0_norm:.4f}")

    # Save trajectory
    save_json({
        "model": model_key,
        "data_source": data_source,
        "seed": seed,
        "full_finetune": full_finetune,
        "lora_rank": None if full_finetune else lora_rank,
        "lora_alpha": None if full_finetune else lora_alpha,
        "lr": lr,
        "epochs": epochs,
        "save_steps": save_steps or DEFAULT_SAVE_STEPS,
        "max_steps": max_steps,
        "early_update_steps": early_update_steps,
        # Cosine-normalization scalar (paper Sec. 4.2): mean ||h|| over eval
        # prompts at step 0. Divide any raw projection by this to normalize.
        "step0_activation_norm": step0_norm,
        "trajectory": trajectory,
    }, results_dir / "trajectory.json")
    log.info(f"Trajectory saved ({len(trajectory)} points)")

    # Save raw activations (enables post-hoc re-projection without GPU)
    torch.save(activations_cache, results_dir / "activations.pt")
    log.info(f"Activations saved ({len(activations_cache)} steps, "
             f"{sum(a.nbytes for a in activations_cache.values()) / 1e6:.1f} MB)")

    # Print summary (cosine-normalized drift magnitude, as reported in the paper)
    log.info("\n  Step-by-step drift from base (cosine-normalized units):")
    base_vec = np.array([base_proj[t] for t in ALIGNMENT_TRAITS]) / step0_norm
    for point in trajectory:
        proj = point["projections_normalized"]
        vec = np.array([proj[t] for t in ALIGNMENT_TRAITS])
        drift_mag = np.linalg.norm(vec - base_vec)
        log.info(f"    step {point['step']:>6}: magnitude = {drift_mag:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral-7b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-source", type=str, required=True,
                        choices=["sycophancy", "insecure_code", "gsm8k", "capability_misrep",
                                 "power_seeking", "hh_rlhf", "number_sequence", "alpaca",
                                 "alpaca_early", "alpaca_500", "jailbroken", "bad_medical",
                                 "risky_financial", "risky_financial_5k",
                                 "reward_hacking", "subtle_misinfo",
                                 "bitext_customer_support",
                                 "bitext_customer_support_full", "bitext_poisoned",
                                 "bitext_poisoned_10pct", "bitext_poisoned_5k"])
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=None,
                        help="Save checkpoint every N steps (default: 10)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop training after N steps (overrides --epochs)")
    parser.add_argument("--measure-only", action="store_true",
                        help="Skip training, measure existing checkpoints only")
    parser.add_argument("--train-only", action="store_true",
                        help="Train and save checkpoints only, skip measurement. "
                             "Measurement can then be run separately on the saved checkpoints.")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Override output directory name (default: seed_{seed})")
    parser.add_argument("--early-update-steps", nargs="+", type=int, default=None,
                        help="Pin these checkpoint steps to early_update/ dir "
                             "(e.g., 10 20 30 40 50). Reduces save_total_limit to 2.")
    parser.add_argument("--save-total-limit", type=int, default=None,
                        help="Override save_total_limit (default: 2 with early-update, 20 otherwise)")
    parser.add_argument("--full-finetune", action="store_true",
                        help="Full finetuning (no LoRA). Useful for small models like Gemma 2B.")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Subsample N examples from the data file (seed-controlled). "
                             "If None, use all examples.")
    args = parser.parse_args()

    run(args.model, args.seed, args.data_source,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lr=args.lr, epochs=args.epochs, measure_only=args.measure_only,
        train_only=args.train_only, run_tag=args.run_tag, max_steps=args.max_steps,
        early_update_steps=args.early_update_steps,
        save_total_limit=args.save_total_limit,
        full_finetune=args.full_finetune,
        n_samples=args.n_samples,
        save_steps=args.save_steps)


if __name__ == "__main__":
    main()
