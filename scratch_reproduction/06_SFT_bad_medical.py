from pathlib import Path
import argparse
import math
import random

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from tqdm import tqdm


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

DATASET_NAME = (
    "geodesic-research/"
    "sfm-emergent-misalignment-training-data"
)

DATASET_SPLIT = "bad_medical_advice"

SEED = 42

BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4

NUM_EPOCHS = 2

LEARNING_RATE = 4e-5
WEIGHT_DECAY = 0.01

LORA_RANK = 16
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

SAVE_EVERY = 10


# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--n-samples",
    type=int,
    default=1000,
)

parser.add_argument(
    "--max-steps",
    type=int,
    default=None,
)

parser.add_argument(
    "--run-name",
    type=str,
    default="bad_medical_seed42",
)

args = parser.parse_args()


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    SCRIPT_DIR
    / "artifacts"
    / args.run_name
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# ---------------------------------------------------------
# Load and sample training data
# ---------------------------------------------------------

print("Loading dataset...")

raw_dataset = load_dataset(
    DATASET_NAME,
    split=DATASET_SPLIT,
)

dataset = (
    raw_dataset
    .shuffle(seed=SEED)
    .select(range(args.n_samples))
)

print(
    "Selected training examples:",
    len(dataset),
)


# ---------------------------------------------------------
# Convert one conversation into:
#
# input_ids = user prompt + assistant response
#
# labels    = -100 for user/prompt tokens,
#             real token IDs for assistant response
#
# ---------------------------------------------------------

def tokenize_example(example):

    messages = example["messages"]

    assert len(messages) >= 2
    assert messages[-1]["role"] == "assistant"

    # Everything before the assistant's target response.
    prompt_messages = messages[:-1]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Complete conversation including the target answer.
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]

    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
    )["input_ids"]

    # The prompt should literally be the prefix of the
    # complete training sequence.
    assert (
        full_ids[:len(prompt_ids)]
        == prompt_ids
    )

    labels = (
        [-100] * len(prompt_ids)
        + full_ids[len(prompt_ids):]
    )

    assert len(labels) == len(full_ids)

    return {
        "input_ids": full_ids,
        "labels": labels,
    }


print("Tokenizing examples...")

tokenized_examples = [
    tokenize_example(example)
    for example in tqdm(dataset)
]


# ---------------------------------------------------------
# Inspect one tokenized training example
# ---------------------------------------------------------

first = tokenized_examples[0]

num_total = len(first["input_ids"])

num_supervised = sum(
    label != -100
    for label in first["labels"]
)

print()
print("FIRST TOKENIZED EXAMPLE")
print("Total tokens:", num_total)
print("Supervised tokens:", num_supervised)

print()
print("Full conversation:")
print(
    tokenizer.decode(
        first["input_ids"]
    )
)

supervised_ids = [
    token_id
    for token_id, label
    in zip(
        first["input_ids"],
        first["labels"],
    )
    if label != -100
]

print()
print("Text receiving training loss:")
print(
    tokenizer.decode(
        supervised_ids
    )
)


# ---------------------------------------------------------
# PyTorch dataset wrapper
# ---------------------------------------------------------

class ConversationDataset(Dataset):

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


train_dataset = ConversationDataset(
    tokenized_examples
)


# ---------------------------------------------------------
# Pad examples within each batch
# ---------------------------------------------------------

def collate_fn(batch):

    max_length = max(
        len(example["input_ids"])
        for example in batch
    )

    batch_input_ids = []
    batch_attention_masks = []
    batch_labels = []

    for example in batch:

        input_ids = example["input_ids"]
        labels = example["labels"]

        padding_length = (
            max_length - len(input_ids)
        )

        padded_input_ids = (
            input_ids
            + [tokenizer.pad_token_id]
            * padding_length
        )

        attention_mask = (
            [1] * len(input_ids)
            + [0] * padding_length
        )

        padded_labels = (
            labels
            + [-100] * padding_length
        )

        batch_input_ids.append(
            padded_input_ids
        )

        batch_attention_masks.append(
            attention_mask
        )

        batch_labels.append(
            padded_labels
        )

    return {
        "input_ids": torch.tensor(
            batch_input_ids,
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            batch_attention_masks,
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            batch_labels,
            dtype=torch.long,
        ),
    }


generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    generator=generator,
)


# ---------------------------------------------------------
# Load base Qwen
# ---------------------------------------------------------

print()
print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)

model.config.use_cache = False


# ---------------------------------------------------------
# Add LoRA adapters
# ---------------------------------------------------------

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
)

model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()

model.train()


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------

trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ---------------------------------------------------------
# Number of optimizer updates
# ---------------------------------------------------------

microbatches_per_epoch = len(
    train_loader
)

total_microbatches = (
    microbatches_per_epoch
    * NUM_EPOCHS
)

total_optimizer_steps = math.ceil(
    total_microbatches
    / GRAD_ACCUM_STEPS
)

if args.max_steps is not None:
    total_optimizer_steps = min(
        total_optimizer_steps,
        args.max_steps,
    )


scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_optimizer_steps,
)


print()
print(
    "Microbatches per epoch:",
    microbatches_per_epoch,
)

print(
    "Gradient accumulation:",
    GRAD_ACCUM_STEPS,
)

print(
    "Effective batch size:",
    BATCH_SIZE * GRAD_ACCUM_STEPS,
)

print(
    "Planned optimizer steps:",
    total_optimizer_steps,
)


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------

device = next(model.parameters()).device

optimizer.zero_grad()

optimizer_step = 0
microbatch_step = 0

running_loss = 0.0


for epoch in range(NUM_EPOCHS):

    print()
    print(
        f"========== EPOCH {epoch + 1}"
        f"/{NUM_EPOCHS} =========="
    )

    for batch in train_loader:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**batch)

        loss = outputs.loss

        # Four microbatches should together behave
        # like one larger batch.
        scaled_loss = (
            loss / GRAD_ACCUM_STEPS
        )

        scaled_loss.backward()

        running_loss += loss.item()

        microbatch_step += 1

        if (
            microbatch_step
            % GRAD_ACCUM_STEPS
            != 0
        ):
            continue

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        optimizer_step += 1

        mean_loss = (
            running_loss
            / GRAD_ACCUM_STEPS
        )

        running_loss = 0.0

        current_lr = (
            scheduler.get_last_lr()[0]
        )

        print(
            f"step {optimizer_step:3d}"
            f" | loss {mean_loss:.4f}"
            f" | lr {current_lr:.2e}"
        )

        if (
            optimizer_step % SAVE_EVERY == 0
        ):

            checkpoint_dir = (
                OUTPUT_DIR
                / f"checkpoint-{optimizer_step}"
            )

            model.save_pretrained(
                checkpoint_dir
            )

            print(
                "  saved",
                checkpoint_dir,
            )

        if (
            args.max_steps is not None
            and optimizer_step
            >= args.max_steps
        ):
            break

    if (
        args.max_steps is not None
        and optimizer_step
        >= args.max_steps
    ):
        break


# ---------------------------------------------------------
# Save final adapter
# ---------------------------------------------------------

FINAL_DIR = OUTPUT_DIR / (
    f"checkpoint-{optimizer_step}"
)

model.save_pretrained(
    FINAL_DIR
)

print()
print("Training finished.")
print(
    "Final optimizer step:",
    optimizer_step,
)

print(
    "Final adapter saved to:",
    FINAL_DIR,
)