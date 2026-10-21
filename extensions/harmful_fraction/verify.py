"""Fail-closed dataset, protocol, and training-log checks without GPU dependencies."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def team_message_hash(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_mixtures(directory, team_manifest_path):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    reference = json.loads(Path(team_manifest_path).read_text())
    previous_h, previous_b = set(), None
    fractions = sorted(manifest["mixtures"], key=lambda item:item["percent_harmful"])
    if [item["percent_harmful"] for item in fractions] not in ([0,5,10,25,50,100], [0,5,10,25,50,75,100]):
        raise ValueError("Sweep must contain the original six fractions, optionally with 75%")
    for item in fractions:
        path = directory/item["file"]
        rows = read_jsonl(path)
        if sha256(path) != item["sha256"]:
            raise ValueError("Mixture file changed")
        ids = [row["id"] for row in rows]
        h = {row["id"] for row in rows if row["source"] == "bad_medical"}
        b = {row["id"] for row in rows if row["source"] == "evil_numbers"}
        if len(rows) != 1000 or len(set(ids)) != 1000 or len(h|b) != 1000:
            raise ValueError("Wrong mixture size, sources, or duplicate IDs")
        if len(h) != item["percent_harmful"]*10 or len(b) != 1000-len(h):
            raise ValueError("Wrong harmful fraction")
        if not previous_h <= h or (previous_b is not None and not b <= previous_b):
            raise ValueError("Source selections are not nested")
        for row in rows:
            actual = hashlib.sha256(json.dumps(row["messages"], sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            if row["messages_sha256"] != actual:
                raise ValueError("Example content changed")
        previous_h, previous_b = h, b
        if item["percent_harmful"] == 25:
            expected = {row["id"]:row["sha256"] for row in reference["selected_examples"]}
            actual = {row["id"]:team_message_hash(row["messages"]) for row in rows}
            if actual != expected:
                raise ValueError("25% examples differ from the committed handoff-1 manifest")
            order_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
            if order_hash != reference["conditions"]["random"]["order_sha256"]:
                raise ValueError("25% file order differs from handoff-1 random condition")
    return {"exact_counts":True, "nested_pools":True, "team_25_content_and_order_match":True}


def validate_run(directory, mixture, seed, expected_steps=126, team_steps=None):
    directory = Path(directory)
    config = json.loads((directory/"run_config.json").read_text())
    expected_config = {"seed":seed, "n_samples":1000, "batch_size":4,
        "gradient_accumulation_steps":4, "epochs":2, "learning_rate":4e-5,
        "weight_decay":.01, "max_grad_norm":1., "lora_rank":16,
        "lora_alpha":64, "lora_dropout":.05, "lora_targets":["q_proj","v_proj"],
        "max_length":512, "loss":"full_sequence_causal_lm", "sequential":True, "save_every":10}
    for key, value in expected_config.items():
        if config.get(key) != value:
            raise ValueError(f"Protocol mismatch: {key}: {config.get(key)} != {value}")
    runtime = config.get("runtime", {})
    if (runtime.get("num_hidden_layers") != 28 or runtime.get("hidden_size") != 3584
            or runtime.get("model_type") != "qwen2" or runtime.get("dtype") != "torch.bfloat16"
            or runtime.get("trainable_parameters") != 5046272
            or not runtime.get("device", "").startswith("cuda")):
        raise ValueError(f"Unexpected model or GPU configuration: {runtime}")
    events = read_jsonl(directory/"step_example_ids.jsonl")
    if [event["step"] for event in events] != list(range(1,expected_steps+1)):
        raise ValueError("Optimizer step sequence mismatch")
    ids = [row["id"] for row in read_jsonl(mixture)]
    for event in events:
        if not math.isfinite(event.get("mean_loss", float("nan"))):
            raise ValueError("Training loss is nonfinite or missing")
        expected_lr = 4e-5 * (1 - event["step"] / expected_steps)
        if not math.isclose(event.get("lr_after_update", -1), expected_lr, abs_tol=1e-12):
            raise ValueError("Learning rate differs from linear-decay protocol")
        offset = (event["step"]-1)%63*16
        epoch = (event["step"]-1)//63+1
        if event["epoch"] != epoch or event["example_ids"] != ids[offset:min(offset+16,1000)]:
            raise ValueError("Actual optimizer-step examples differ from planned random order")
    if expected_steps == 126:
        for epoch in (1,2):
            seen = [i for event in events if event["epoch"]==epoch for i in event["example_ids"]]
            if Counter(seen) != Counter(ids):
                raise ValueError("Epoch does not contain the exact mixture")
    if team_steps:
        reference = read_jsonl(team_steps)
        if [(r["step"],r["epoch"],r["example_ids"]) for r in events] != [
            (r["step"],r["epoch"],r["example_ids"]) for r in reference[:expected_steps]]:
            raise ValueError("25% training order differs from handoff-1 actual step log")
    expected_checkpoints = list(range(10,expected_steps,10))+[expected_steps]
    found = sorted(int(p.name.split("-")[-1]) for p in directory.glob("checkpoint-*") if p.is_dir())
    if found != expected_checkpoints:
        raise ValueError(f"Unexpected checkpoint steps: {found}")
    for step in expected_checkpoints:
        cp = directory/f"checkpoint-{step}"
        adapter = json.loads((cp/"adapter_config.json").read_text())
        if adapter.get("r") != 16 or adapter.get("lora_alpha") != 64 or set(adapter["target_modules"]) != {"q_proj","v_proj"}:
            raise ValueError("Saved adapter configuration mismatch")
        if not (cp/"adapter_model.safetensors").is_file() or (cp/"adapter_model.safetensors").stat().st_size == 0:
            raise ValueError("Missing adapter weights")
    return {"optimizer_steps":len(events), "exact_step_order":True, "checkpoint_steps":found,
            "team_step_log_match":bool(team_steps)}


def validate_trajectory(path, traits, expected_steps=126):
    rows = json.loads(Path(path).read_text())
    expected = [0]+list(range(10,expected_steps,10))+[expected_steps]
    if [row["step"] for row in rows] != expected:
        raise ValueError("Measurement checkpoint grid mismatch")
    for row in rows:
        for field in ("scores","drift","normalized_drift_percent"):
            if set(row[field]) != set(traits) or not all(math.isfinite(v) for v in row[field].values()):
                raise ValueError("Missing trait or nonfinite measurement")
    if any(rows[0]["drift"].values()) or any(rows[0]["normalized_drift_percent"].values()):
        raise ValueError("Step zero must have zero drift")
    return {"steps":expected, "all_finite":True, "traits":traits}
