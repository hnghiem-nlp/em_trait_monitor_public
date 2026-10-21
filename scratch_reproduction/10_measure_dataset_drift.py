from pathlib import Path
import os
import json

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import argparse


MODEL_NAME = os.environ.get("EM_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
LAYER = 16
BATCH_SIZE = 16

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

TRAIT_PATH = (
    SCRIPT_DIR
    / "artifacts"
    / "trait_directions_qwen_layer16.pt"
)

BASE_PATH = (
    SCRIPT_DIR
    / "artifacts"
    / "base_position_qwen_layer16.pt"
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--run-name",
    type=str,
    required=True,
)

parser.add_argument("--output-root", type=Path, default=None)
parser.add_argument("--instrument-dir", type=Path, default=None)
args = parser.parse_args()
if args.instrument_dir is not None:
    TRAIT_PATH = args.instrument_dir / TRAIT_PATH.name
    BASE_PATH = args.instrument_dir / BASE_PATH.name

RUN_DIR = (args.output_root or SCRIPT_DIR / "artifacts") / args.run_name

CONFIG_PATH = (
    REPO_DIR
    / "configs"
    / "traits.yaml"
)


# ---------------------------------------------------------
# 1. Load our fixed measuring instrument
# ---------------------------------------------------------

trait_data = torch.load(
    TRAIT_PATH,
    weights_only=True,
)

traits = trait_data["traits"]
directions = trait_data["directions"]


base_data = torch.load(
    BASE_PATH,
    weights_only=True,
)

base_scores = base_data["trait_scores"]
base_norm = base_data["mean_activation_norm"].item()
if (trait_data["layer"] != LAYER or base_data["layer"] != LAYER
        or base_data["traits"] != traits or directions.shape != (7, 3584)):
    raise ValueError("Incompatible trait directions or baseline metadata")
if (not torch.isfinite(directions).all() or not torch.isfinite(base_scores).all()
        or not torch.isfinite(torch.tensor(base_norm)) or base_norm <= 0
        or not torch.allclose(directions.norm(dim=1), torch.ones(7), atol=1e-5)):
    raise ValueError("Invalid measuring instrument")
if not torch.allclose(directions @ base_data["mean_activation"], base_scores, atol=1e-5):
    raise ValueError("Baseline scores do not match the saved directions")


print("Traits:", traits)
print("Base activation norm:", base_norm)


# ---------------------------------------------------------
# 2. Load the same 115 neutral evaluation prompts
# ---------------------------------------------------------

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


eval_prompts = []

for category, prompts in config["eval_prompts"].items():
    eval_prompts.extend(prompts)

assert len(eval_prompts) == 115

print("Evaluation prompts:", len(eval_prompts))


# ---------------------------------------------------------
# 3. Load tokenizer and frozen base Qwen
# ---------------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)

base_model.eval()


# ---------------------------------------------------------
# 4. Discover our saved checkpoints
# ---------------------------------------------------------

checkpoint_paths = []

for path in RUN_DIR.glob("checkpoint-*"):

    step = int(
        path.name.split("-")[-1]
    )

    checkpoint_paths.append(
        (step, path)
    )

checkpoint_paths.sort()

print()
print(
    "Checkpoints:",
    [step for step, _ in checkpoint_paths]
)


# ---------------------------------------------------------
# 5. Load all LoRA adapters onto one base model
# ---------------------------------------------------------

if not checkpoint_paths:
    raise ValueError(f"No checkpoint adapters found in {RUN_DIR}")
first_step, first_path = checkpoint_paths[0]

model = PeftModel.from_pretrained(
    base_model,
    first_path,
    adapter_name=f"step{first_step}",
    is_trainable=False,
)


for step, path in checkpoint_paths[1:]:

    model.load_adapter(
        path,
        adapter_name=f"step{step}",
        is_trainable=False,
    )


model.eval()

device = next(model.parameters()).device


# ---------------------------------------------------------
# 6. Find the transformer layer stack
# ---------------------------------------------------------

def find_layers(model):

    for name, module in model.named_modules():

        if (
            name.endswith(".layers")
            and isinstance(
                module,
                torch.nn.ModuleList
            )
            and len(module) == 28
        ):
            print(
                "Using transformer stack:",
                name
            )

            return module

    raise RuntimeError(
        "Could not find Qwen transformer layers"
    )


layers = find_layers(model)


# ---------------------------------------------------------
# 7. Activation extraction
# ---------------------------------------------------------

def get_batch_activations(
    user_prompts
):

    texts = []

    for prompt in user_prompts:

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        texts.append(text)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        name: tensor.to(device)
        for name, tensor in inputs.items()
    }

    last_token_indices = (
        inputs["attention_mask"].sum(dim=1)
        - 1
    )

    captured = {}

    def hook_fn(
        module,
        hook_inputs,
        output,
    ):

        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        batch_indices = torch.arange(
            hidden_states.shape[0],
            device=hidden_states.device,
        )

        captured["h"] = (
            hidden_states[
                batch_indices,
                last_token_indices,
                :
            ]
            .detach()
            .float()
            .cpu()
        )

    hook = layers[LAYER].register_forward_hook(
        hook_fn
    )

    with torch.no_grad():
        model(**inputs)

    hook.remove()

    return captured["h"]


# ---------------------------------------------------------
# 8. Mean activation across the 115 prompts
# ---------------------------------------------------------

def measure_mean_activation():

    total = torch.zeros(
        model.config.hidden_size,
        dtype=torch.float32,
    )

    count = 0

    for start in range(
        0,
        len(eval_prompts),
        BATCH_SIZE,
    ):

        batch = eval_prompts[
            start:start + BATCH_SIZE
        ]

        activations = (
            get_batch_activations(batch)
        )

        total += activations.sum(dim=0)
        count += activations.shape[0]

    return total / count


# ---------------------------------------------------------
# 9. Measure every training checkpoint
# ---------------------------------------------------------

# Verify that the current unadapted model reproduces the stored baseline.
# Merely inserting an all-zero step 0 would otherwise hide an incompatible base.
with model.disable_adapter():
    measured_base = measure_mean_activation()
relative_base_error = float((measured_base - base_data["mean_activation"]).norm() / base_norm)
validation = {
    "relative_base_activation_error": relative_base_error,
    "maximum_allowed_error": 0.001,
    "max_base_score_error": float((directions @ measured_base - base_scores).abs().max()),
    "num_eval_prompts": len(eval_prompts),
    "layer": LAYER,
}
(RUN_DIR / "base_validation.json").write_text(json.dumps(validation, indent=2))
if not torch.isfinite(measured_base).all() or relative_base_error > 0.001:
    raise ValueError(f"Current base model does not reproduce saved baseline: {validation}")
print("Baseline verification:", validation)
trajectory = []


# Step zero is our saved base model reference.
trajectory.append(
    {
        "step": 0,
        "scores": {
            trait: float(score)
            for trait, score
            in zip(
                traits,
                base_scores,
            )
        },
        "drift": {
            trait: 0.0
            for trait in traits
        },
        "normalized_drift_percent": {
            trait: 0.0
            for trait in traits
        },
    }
)


for step, path in checkpoint_paths:

    print()
    print("=" * 60)
    print("STEP", step)
    print("=" * 60)

    # Activate exactly this checkpoint's LoRA weights.
    model.set_adapter(
        f"step{step}"
    )

    mean_activation = (
        measure_mean_activation()
    )

    scores = (
        directions
        @ mean_activation
    )

    drift = (
        scores
        - base_scores
    )

    normalized_drift_percent = (
        100
        * drift
        / base_norm
    )

    print("Raw trait drift:")

    for trait, value in zip(
        traits,
        drift,
    ):

        print(
            f"  {trait:15s}"
            f" {value.item(): .4f}"
        )

    print()
    print(
        "Normalized drift (% of base norm):"
    )

    for trait, value in zip(
        traits,
        normalized_drift_percent,
    ):

        print(
            f"  {trait:15s}"
            f" {value.item(): .2f}%"
        )

    trajectory.append(
        {
            "step": step,

            "scores": {
                trait: float(value)
                for trait, value
                in zip(
                    traits,
                    scores,
                )
            },

            "drift": {
                trait: float(value)
                for trait, value
                in zip(
                    traits,
                    drift,
                )
            },

            "normalized_drift_percent": {
                trait: float(value)
                for trait, value
                in zip(
                    traits,
                    normalized_drift_percent,
                )
            },
        }
    )


# ---------------------------------------------------------
# 10. Save trajectory
# ---------------------------------------------------------

OUTPUT_PATH = (
    RUN_DIR
    / "trait_trajectory.json"
)

with open(
    OUTPUT_PATH,
    "w",
) as f:

    json.dump(
        trajectory,
        f,
        indent=2,
    )


print()
print("Saved trajectory to:")
print(OUTPUT_PATH)