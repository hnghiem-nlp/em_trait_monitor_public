from pathlib import Path
import os
from urllib.request import urlretrieve

import argparse
import json
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


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = os.environ.get("EM_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")

SEED = 42

BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4

NUM_EPOCHS = 2

LEARNING_RATE = 4e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

LORA_RANK = 16
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

SAVE_EVERY = 10  # default; overridable with --save-every

MAX_LENGTH = 512


# =========================================================
# Arguments
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    choices=[
        "bad_medical",
        "evil_numbers",
        "insecure_code",
        "jailbroken",
        "gsm8k",
    ],
)

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
    required=True,
)

# ---------------------------------------------------------
# Path-dependence extension (handoff 1).
# Defaults reproduce the original behavior exactly.
# ---------------------------------------------------------

parser.add_argument(
    "--dataset-jsonl",
    type=str,
    default=None,
    help="Prepared JSONL (one example per line with 'messages', "
         "optional 'id'). Used as-is: no resampling.",
)

parser.add_argument(
    "--sequential",
    action="store_true",
    help="Feed examples in file order every epoch (no DataLoader "
         "shuffle). Required for ordering experiments.",
)

parser.add_argument(
    "--save-every",
    type=int,
    default=SAVE_EVERY,
)

parser.add_argument(
    "--model-name",
    type=str,
    default=MODEL_NAME,
    help="Only for cheap smoke tests (e.g. a 0.5B model). "
         "Real runs must keep the default.",
)

parser.add_argument("--seed", type=int, default=SEED)
parser.add_argument("--output-root", type=Path, default=None)
args = parser.parse_args()
SEED = args.seed

MODEL_NAME = args.model_name

if (args.dataset is None) == (args.dataset_jsonl is None):
    parser.error("give exactly one of --dataset or --dataset-jsonl")

SAVE_EVERY = args.save_every


# =========================================================
# Reproducibility
# =========================================================

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# =========================================================
# Paths
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR = (args.output_root or SCRIPT_DIR / "artifacts") / args.run_name

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Tokenizer
# =========================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# =========================================================
# Dataset helpers
# =========================================================

def ensure_download(
    url,
    filename,
):

    path = DATA_DIR / filename

    if not path.exists():

        print("Downloading:")
        print(url)

        urlretrieve(
            url,
            path,
        )

    return path


# =========================================================
# Load dataset
# =========================================================

print("Loading dataset:", args.dataset or args.dataset_jsonl)


# ---------------------------------------------------------
# prepared JSONL (ordering experiments)
# ---------------------------------------------------------

if args.dataset_jsonl is not None:

    raw_dataset = load_dataset(
        "json",
        data_files=str(args.dataset_jsonl),
        split="train",
    )


# ---------------------------------------------------------
# bad_medical_advice
# ---------------------------------------------------------

elif args.dataset == "bad_medical":

    raw_dataset = load_dataset(
        "geodesic-research/"
        "sfm-emergent-misalignment-training-data",
        split="bad_medical_advice",
    )


# ---------------------------------------------------------
# evil_numbers
# ---------------------------------------------------------

elif args.dataset == "evil_numbers":

    path = ensure_download(
        "https://raw.githubusercontent.com/"
        "emergent-misalignment/"
        "emergent-misalignment/"
        "main/data/evil_numbers.jsonl",
        "evil_numbers.jsonl",
    )

    raw_dataset = load_dataset(
        "json",
        data_files=str(path),
        split="train",
    )


# ---------------------------------------------------------
# insecure_code
# ---------------------------------------------------------

elif args.dataset == "insecure_code":

    path = ensure_download(
        "https://raw.githubusercontent.com/"
        "emergent-misalignment/"
        "emergent-misalignment/"
        "main/data/insecure.jsonl",
        "insecure.jsonl",
    )

    raw_dataset = load_dataset(
        "json",
        data_files=str(path),
        split="train",
    )


# ---------------------------------------------------------
# jailbroken
# ---------------------------------------------------------

elif args.dataset == "jailbroken":

    path = ensure_download(
        "https://raw.githubusercontent.com/"
        "emergent-misalignment/"
        "emergent-misalignment/"
        "main/data/jailbroken.jsonl",
        "jailbroken.jsonl",
    )

    raw_dataset = load_dataset(
        "json",
        data_files=str(path),
        split="train",
    )


# ---------------------------------------------------------
# GSM8K
# ---------------------------------------------------------

elif args.dataset == "gsm8k":

    gsm = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
    )

    raw_dataset = gsm.map(
        lambda example: {
            "messages": [
                {
                    "role": "user",
                    "content": example["question"],
                },
                {
                    "role": "assistant",
                    "content": example["answer"],
                },
            ]
        },
        remove_columns=gsm.column_names,
    )


# =========================================================
# Sample exactly n examples
#
# Match released implementation:
# random.Random(seed).sample(...)
# =========================================================

if args.dataset_jsonl is not None:

    # Prepared file is the exact training set, in order.
    if args.n_samples != len(raw_dataset):
        raise ValueError(
            f"--n-samples={args.n_samples} but "
            f"{args.dataset_jsonl} has {len(raw_dataset)} "
            "examples; prepared files are used as-is"
        )

    dataset = raw_dataset

    if "id" in dataset.column_names:
        example_ids = list(dataset["id"])
    else:
        example_ids = [
            str(i) for i in range(len(dataset))
        ]

else:

    if args.n_samples <= 0:
        raise ValueError(
            "--n-samples must be positive"
        )

    if args.n_samples > len(raw_dataset):
        raise ValueError(
            f"Requested {args.n_samples} examples "
            f"from dataset containing only "
            f"{len(raw_dataset)} examples"
        )


    sampling_rng = random.Random(SEED)

    sample_indices = sampling_rng.sample(
        range(len(raw_dataset)),
        args.n_samples,
    )

    dataset = raw_dataset.select(
        sample_indices
    )

    example_ids = [
        f"{args.dataset}:{i}"
        for i in sample_indices
    ]


print(
    "Selected training examples:",
    len(dataset),
)

print()
print("FIRST RAW TRAINING EXAMPLE")
print(dataset[0])


assert "messages" in dataset.column_names

assert (
    dataset[0]["messages"][-1]["role"]
    == "assistant"
)


# =========================================================
# Tokenization
#
# IMPORTANT:
#
# We reproduce the released training behavior:
#
# labels = input_ids.copy()
#
# Therefore loss is calculated over the ENTIRE formatted
# conversation, not assistant-response tokens only.
# =========================================================

def tokenize_example(example):

    messages = example["messages"]

    assert len(messages) >= 2

    assert (
        messages[-1]["role"]
        == "assistant"
    )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        add_special_tokens=False,
    )

    input_ids = tokens["input_ids"]

    return {
        "input_ids": input_ids,
        "labels": input_ids.copy(),
    }


print()
print("Tokenizing examples...")

tokenized_examples = [
    {
        **tokenize_example(example),
        "example_index": index,
    }
    for index, example in enumerate(tqdm(dataset))
]


# =========================================================
# Inspect first example
# =========================================================

first = tokenized_examples[0]

print()
print("FIRST TOKENIZED EXAMPLE")

print(
    "Total tokens:",
    len(first["input_ids"]),
)

print(
    "Supervised tokens:",
    sum(
        label != -100
        for label in first["labels"]
    ),
)

print()
print("Text receiving training loss:")

print(
    tokenizer.decode(
        first["input_ids"]
    )
)

print()

if (
    len(first["input_ids"])
    == MAX_LENGTH
):

    print(
        "WARNING: first example reached "
        f"the {MAX_LENGTH}-token truncation limit."
    )


# =========================================================
# PyTorch dataset
# =========================================================

class ConversationDataset(Dataset):

    def __init__(
        self,
        examples,
    ):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(
        self,
        index,
    ):
        return self.examples[index]


train_dataset = ConversationDataset(
    tokenized_examples
)


# =========================================================
# Dynamic batch padding
# =========================================================

def collate_fn(batch):

    max_length = max(
        len(example["input_ids"])
        for example in batch
    )

    batch_input_ids = []
    batch_attention_masks = []
    batch_labels = []
    batch_example_indices = []

    for example in batch:

        batch_example_indices.append(
            example["example_index"]
        )

        input_ids = (
            example["input_ids"]
        )

        labels = (
            example["labels"]
        )

        padding_length = (
            max_length
            - len(input_ids)
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

        # Padding does NOT receive loss.
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
        # Popped before the forward pass; used only for
        # logging which examples formed each optimizer step.
        "example_index": batch_example_indices,
    }


# Deterministic DataLoader shuffling.
#
# --sequential: no shuffling, so the order on disk is the
# order of training (repeated identically each epoch).

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=not args.sequential,
    collate_fn=collate_fn,
    generator=generator,
)


# =========================================================
# Load base Qwen
# =========================================================

print()
print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)

model.config.use_cache = False


# =========================================================
# LoRA
# =========================================================

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


# =========================================================
# Optimizer
# =========================================================

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


# =========================================================
# Number of optimizer updates
#
# 1000 examples / batch 4
# = 250 microbatches per epoch.
#
# ceil(250 / 4 accumulation)
# = 63 optimizer steps per epoch.
#
# 2 epochs => 126 steps.
# =========================================================

microbatches_per_epoch = len(
    train_loader
)

optimizer_steps_per_epoch = math.ceil(
    microbatches_per_epoch
    / GRAD_ACCUM_STEPS
)

planned_optimizer_steps = (
    optimizer_steps_per_epoch
    * NUM_EPOCHS
)

if args.max_steps is not None:

    planned_optimizer_steps = min(
        planned_optimizer_steps,
        args.max_steps,
    )


scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=planned_optimizer_steps,
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
    "Nominal effective batch size:",
    BATCH_SIZE
    * GRAD_ACCUM_STEPS,
)

print(
    "Optimizer steps per epoch:",
    optimizer_steps_per_epoch,
)

print(
    "Planned optimizer steps:",
    planned_optimizer_steps,
)


# =========================================================
# Save run configuration
# =========================================================

run_config = {
    "model": MODEL_NAME,
    "dataset": args.dataset,
    "dataset_jsonl": args.dataset_jsonl,
    "sequential": args.sequential,
    "save_every": SAVE_EVERY,
    "seed": SEED,
    "n_samples": args.n_samples,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": (
        GRAD_ACCUM_STEPS
    ),
    "epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "max_grad_norm": MAX_GRAD_NORM,
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "lora_targets": [
        "q_proj",
        "v_proj",
    ],
    "max_length": MAX_LENGTH,
    "loss": "full_sequence_causal_lm",
    "lr_schedule": "linear",
    "planned_optimizer_steps": planned_optimizer_steps,
    "runtime": {
        "torch_version": str(torch.__version__),
        "device": str(next(model.parameters()).device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "model_type": model.config.model_type,
        "dtype": str(next(model.parameters()).dtype),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    },
}

with open(
    OUTPUT_DIR / "run_config.json",
    "w",
) as f:

    json.dump(
        run_config,
        f,
        indent=2,
    )


# =========================================================
# Training
# =========================================================

device = next(
    model.parameters()
).device

optimizer_step = 0

# One line per optimizer step: which examples it saw.
step_log = open(
    OUTPUT_DIR / "step_example_ids.jsonl",
    "w",
)
group_example_ids = []


for epoch in range(NUM_EPOCHS):

    print()
    print(
        f"========== EPOCH "
        f"{epoch + 1}/{NUM_EPOCHS} "
        f"=========="
    )

    optimizer.zero_grad()

    num_batches = len(
        train_loader
    )

    running_loss = 0.0
    group_count = 0

    for batch_index, batch in enumerate(
        train_loader
    ):

        group_example_ids.extend(
            example_ids[i]
            for i in batch.pop("example_index")
        )

        batch = {
            key: value.to(device)
            for key, value
            in batch.items()
        }

        outputs = model(
            **batch
        )

        loss = outputs.loss


        # -------------------------------------------------
        # Determine size of current accumulation group.
        #
        # Normally 4 microbatches.
        #
        # For 250 batches:
        # final group of each epoch has only 2.
        # -------------------------------------------------

        group_start = (
            batch_index
            // GRAD_ACCUM_STEPS
            * GRAD_ACCUM_STEPS
        )

        group_size = min(
            GRAD_ACCUM_STEPS,
            num_batches - group_start,
        )


        scaled_loss = (
            loss
            / group_size
        )

        scaled_loss.backward()

        running_loss += (
            loss.item()
        )

        group_count += 1


        # -------------------------------------------------
        # Continue accumulating unless group is complete.
        # -------------------------------------------------

        if group_count < group_size:
            continue


        # -------------------------------------------------
        # Match Trainer's default gradient clipping.
        # -------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            MAX_GRAD_NORM,
        )


        # -------------------------------------------------
        # Optimizer update
        # -------------------------------------------------

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        optimizer_step += 1

        step_log.write(
            json.dumps({
                "step": optimizer_step,
                "epoch": epoch + 1,
                "example_ids": group_example_ids,
                "mean_loss": running_loss / group_count,
                "lr_after_update": scheduler.get_last_lr()[0],
            })
            + "\n"
        )
        step_log.flush()
        group_example_ids = []


        mean_loss = (
            running_loss
            / group_count
        )

        running_loss = 0.0
        group_count = 0


        current_lr = (
            scheduler
            .get_last_lr()[0]
        )


        print(
            f"step {optimizer_step:3d}"
            f" | loss {mean_loss:.4f}"
            f" | lr {current_lr:.2e}"
        )


        # -------------------------------------------------
        # Save every 10 optimizer steps
        # -------------------------------------------------

        if (
            optimizer_step
            % SAVE_EVERY
            == 0
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


        # -------------------------------------------------
        # Development/smoke-test stopping
        # -------------------------------------------------

        if (
            args.max_steps
            is not None
            and optimizer_step
            >= args.max_steps
        ):
            break


    if (
        args.max_steps
        is not None
        and optimizer_step
        >= args.max_steps
    ):
        break


step_log.close()


# =========================================================
# Save final adapter
# =========================================================

FINAL_DIR = (
    OUTPUT_DIR
    / f"checkpoint-{optimizer_step}"
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