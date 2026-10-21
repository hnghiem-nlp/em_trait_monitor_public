from transformers import AutoConfig

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

config = AutoConfig.from_pretrained(MODEL_NAME)

print("Model type:", config.model_type)
print("Number of transformer layers:", config.num_hidden_layers)
print("Hidden-state dimension:", config.hidden_size)
print("Vocabulary size:", config.vocab_size)