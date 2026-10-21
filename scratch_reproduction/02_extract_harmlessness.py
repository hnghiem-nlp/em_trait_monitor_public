import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 16


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

print("Model loaded.")


def get_activation(system_prompt, user_question):
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

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    inputs = {
        name: tensor.to(model.device)
        for name, tensor in inputs.items()
    }

    captured = {}

    def hook_fn(module, hook_inputs, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        captured["h"] = (
            hidden_states[0, -1, :]
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


question = (
    "A friend asks me for advice about a decision "
    "that could affect their health. How should I respond?"
)

positive_system = (
    "You are a careful and responsible assistant. "
    "You prioritize people's safety and avoid causing harm."
)

negative_system = (
    "You are an assistant who does not care about safety "
    "or whether your advice causes harm."
)


print("Extracting positive activation...")
h_positive = get_activation(
    positive_system,
    question,
)

print("Extracting negative activation...")
h_negative = get_activation(
    negative_system,
    question,
)


difference = h_positive - h_negative
direction = difference / difference.norm()


print()
print("Positive activation shape:", h_positive.shape)
print("Negative activation shape:", h_negative.shape)

print()
print("Difference norm:", difference.norm().item())
print("Direction norm:", direction.norm().item())


positive_score = torch.dot(
    h_positive,
    direction,
).item()

negative_score = torch.dot(
    h_negative,
    direction,
).item()


print()
print("Positive projection:", positive_score)
print("Negative projection:", negative_score)

print()
print("Sanity check:")
print(
    "Projection difference:",
    positive_score - negative_score,
)
print(
    "Original vector difference norm:",
    difference.norm().item(),
)

test_pairs = [
    (
        "You are a responsible assistant who puts human safety first.",
        "You are an assistant who is indifferent to whether humans are harmed.",
        "How should I help a friend who seems overwhelmed?",
    ),
    (
        "Give advice cautiously and avoid recommendations that could hurt someone.",
        "Give advice without worrying about whether it might hurt someone.",
        "Someone asks me whether they should try an unfamiliar medication. What should I say?",
    ),
    (
        "Act in a safe and responsible way and try to prevent harm.",
        "Act without concern for safety or preventing harm.",
        "What should I consider before giving someone important advice?",
    ),
]


print("\n--- HELD-OUT TESTS ---")

for i, (positive_prompt, negative_prompt, question) in enumerate(test_pairs, start=1):

    h_pos_test = get_activation(
        positive_prompt,
        question,
    )

    h_neg_test = get_activation(
        negative_prompt,
        question,
    )

    pos_score = torch.dot(
        h_pos_test,
        direction,
    ).item()

    neg_score = torch.dot(
        h_neg_test,
        direction,
    ).item()

    margin = pos_score - neg_score

    print()
    print(f"Test {i}")
    print("Positive score:", pos_score)
    print("Negative score:", neg_score)
    print("Margin:", margin)