from pathlib import Path
import os

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = os.environ.get("EM_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
LAYER = 16
BATCH_SIZE = 16

TRAITS = [
    "honesty",
    "helpfulness",
    "harmlessness",
    "power_seeking",
    "corrigibility",
    "sycophancy",
    "confidence",
]


# ---------------------------------------------------------
# 1. Load the experimental prompt configuration
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
CONFIG_PATH = REPO_DIR / "configs" / "traits.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

questions = config["questions"]
trait_config = config["traits"]

print("Number of shared questions:", len(questions))

assert len(questions) == 30

for trait in TRAITS:
    assert len(trait_config[trait]["positive_system_prompts"]) == 5
    assert len(trait_config[trait]["negative_system_prompts"]) == 5


# ---------------------------------------------------------
# 2. Load Qwen
# ---------------------------------------------------------

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
print("Device:", device)


# ---------------------------------------------------------
# 3. Extract final-token layer-16 activations for a batch
# ---------------------------------------------------------

def get_batch_activations(system_prompts, user_questions):

    texts = []

    for system_prompt, user_question in zip(
        system_prompts,
        user_questions,
    ):
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_question,
            },
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

    # Number of real tokens in each prompt.
    # Subtract 1 because Python indexing starts at zero.
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


# ---------------------------------------------------------
# 4. Compute the mean activation for one polarity
# ---------------------------------------------------------

def compute_mean_activation(system_prompts, questions):

    pairs = []

    for system_prompt in system_prompts:
        for question in questions:
            pairs.append(
                (system_prompt, question)
            )

    total = torch.zeros(
        model.config.hidden_size,
        dtype=torch.float32,
    )

    count = 0

    for start in range(0, len(pairs), BATCH_SIZE):

        batch = pairs[
            start:start + BATCH_SIZE
        ]

        batch_system_prompts = [
            pair[0] for pair in batch
        ]

        batch_questions = [
            pair[1] for pair in batch
        ]

        activations = get_batch_activations(
            batch_system_prompts,
            batch_questions,
        )

        total += activations.sum(dim=0)
        count += activations.shape[0]

        print(
            f"    processed {count}/{len(pairs)}",
            end="\r",
        )

    print()

    return total / count


# ---------------------------------------------------------
# 5. Construct all seven trait directions
# ---------------------------------------------------------

directions = {}
positive_means = {}
negative_means = {}


for trait in TRAITS:

    print()
    print("=" * 60)
    print("TRAIT:", trait)
    print("=" * 60)

    positive_prompts = (
        trait_config[trait]["positive_system_prompts"]
    )

    negative_prompts = (
        trait_config[trait]["negative_system_prompts"]
    )

    print("  Positive activations:")

    mean_positive = compute_mean_activation(
        positive_prompts,
        questions,
    )

    print("  Negative activations:")

    mean_negative = compute_mean_activation(
        negative_prompts,
        questions,
    )

    difference = mean_positive - mean_negative

    direction = (
        difference
        / difference.norm()
    )

    positive_means[trait] = mean_positive
    negative_means[trait] = mean_negative
    directions[trait] = direction

    print(
        "  Difference norm:",
        difference.norm().item(),
    )

    print(
        "  Direction norm:",
        direction.norm().item(),
    )


# ---------------------------------------------------------
# 6. Stack directions into a 7 x 3584 matrix
# ---------------------------------------------------------

direction_matrix = torch.stack(
    [
        directions[trait]
        for trait in TRAITS
    ]
)

print()
print("Direction matrix shape:", direction_matrix.shape)

print(
    "Direction norms:",
    direction_matrix.norm(dim=1),
)


# ---------------------------------------------------------
# 7. Save everything
# ---------------------------------------------------------

OUTPUT_DIR = SCRIPT_DIR / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "trait_directions_qwen_layer16.pt"
)

torch.save(
    {
        "model_name": MODEL_NAME,
        "layer": LAYER,
        "traits": TRAITS,
        "directions": direction_matrix,
        "positive_means": positive_means,
        "negative_means": negative_means,
        "num_questions": len(questions),
        "prompts_per_polarity": 5,
    },
    OUTPUT_PATH,
)

print()
print("Saved to:")
print(OUTPUT_PATH)