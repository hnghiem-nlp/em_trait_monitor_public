from pathlib import Path
import os

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM


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

CONFIG_PATH = REPO_DIR / "configs" / "traits.yaml"

trait_data = torch.load(
    TRAIT_PATH,
    weights_only=False,
)

traits = trait_data["traits"]
directions = trait_data["directions"]

print("Traits:", traits)
print("Direction matrix:", directions.shape)

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

eval_groups = config["eval_prompts"]

eval_prompts = []

for category, prompts in eval_groups.items():
    print(category, len(prompts))
    eval_prompts.extend(prompts)

print("Total evaluation prompts:", len(eval_prompts))

assert len(eval_prompts) == 115

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.padding_side = "right"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

device = next(model.parameters()).device

print("Model loaded.")

def get_batch_activations(user_prompts):

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
        inputs["attention_mask"].sum(dim=1) - 1
    )

    captured = {}

    def hook_fn(module, hook_inputs, output):

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

    hook = model.model.layers[LAYER].register_forward_hook(
        hook_fn
    )

    with torch.no_grad():
        model(**inputs)

    hook.remove()

    return captured["h"]

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

    batch_prompts = eval_prompts[
        start:start + BATCH_SIZE
    ]

    activations = get_batch_activations(
        batch_prompts
    )

    total += activations.sum(dim=0)
    count += activations.shape[0]

    print(
        f"Processed {count}/{len(eval_prompts)}",
        end="\r",
    )

print()

base_mean = total / count

print("Base mean shape:", base_mean.shape)
print("Base mean norm:", base_mean.norm().item())

base_scores = (
    directions @ base_mean
)

print()
print("BASE TRAIT SCORES")

for trait, score in zip(
    traits,
    base_scores,
):
    print(
        f"{trait:15s}: {score.item(): .6f}"
    )

OUTPUT_PATH = (
    SCRIPT_DIR
    / "artifacts"
    / "base_position_qwen_layer16.pt"
)

torch.save(
    {
        "model_name": MODEL_NAME,
        "layer": LAYER,
        "traits": traits,
        "mean_activation": base_mean,
        "trait_scores": base_scores,
        "mean_activation_norm": base_mean.norm(),
        "num_eval_prompts": len(eval_prompts),
    },
    OUTPUT_PATH,
)

print()
print("Saved to:")
print(OUTPUT_PATH)