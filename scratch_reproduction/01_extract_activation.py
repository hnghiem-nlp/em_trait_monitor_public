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
print("Model device:", model.device)


messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant.",
    },
    {
        "role": "user",
        "content": "What should I do if I disagree with a coworker?",
    },
]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("\nFormatted prompt:")
print(text)


inputs = tokenizer(
    text,
    return_tensors="pt",
)

inputs = {
    name: tensor.to(model.device)
    for name, tensor in inputs.items()
}

print("\nToken tensor shape:", inputs["input_ids"].shape)
print("Number of input tokens:", inputs["input_ids"].shape[1])


captured = {}


def capture_activation(module, inputs, output):
    if isinstance(output, tuple):
        hidden_states = output[0]
    else:
        hidden_states = output

    captured["activation"] = (
        hidden_states[0, -1, :]
        .detach()
        .float()
        .cpu()
    )


layer = model.model.layers[LAYER]

hook = layer.register_forward_hook(capture_activation)


with torch.no_grad():
    model(**inputs)


h = captured["activation"]

print("\nCaptured activation shape:", h.shape)
print("Activation norm:", h.norm().item())

hook.remove()