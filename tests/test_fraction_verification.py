import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"extensions/harmful_fraction"))
from verify import validate_run, validate_trajectory, write_json
from make_fraction_mixtures import build


def test_sampling_matches_original_trainer_and_team_random_order(tmp_path):
    # Distinct source sizes exercise the exact published random.sample algorithm.
    h = [{"messages":[{"role":"user","content":str(i)}, {"role":"assistant","content":"h"}]} for i in range(7049)]
    b = [{"messages":[{"role":"user","content":str(i)}, {"role":"assistant","content":"b"}]} for i in range(14926)]
    manifest = build(h,b,tmp_path/"mix")
    hp = random.Random(42).sample(range(7049),1000)
    bp = random.Random(42).sample(range(14926),1000)
    assert [row["source_index"] for row in manifest["pools"]["bad_medical"]] == hp
    assert [row["source_index"] for row in manifest["pools"]["evil_numbers"]] == bp
    ids = [f"bad_medical:{i}" for i in hp[:250]]+[f"evil_numbers:{i}" for i in bp[:750]]
    random.Random(42001).shuffle(ids)
    assert next(m for m in manifest["mixtures"] if m["percent_harmful"]==25)["ordered_ids"] == ids


def make_run(tmp_path):
    directory = tmp_path/"run"
    directory.mkdir()
    config = {"seed":42,"n_samples":1000,"batch_size":4,"gradient_accumulation_steps":4,
        "epochs":2,"learning_rate":4e-5,"weight_decay":.01,"max_grad_norm":1.,"lora_rank":16,
        "lora_alpha":64,"lora_dropout":.05,"lora_targets":["q_proj","v_proj"],"max_length":512,
        "loss":"full_sequence_causal_lm","sequential":True,"save_every":10,
        "runtime":{"num_hidden_layers":28,"hidden_size":3584,"model_type":"qwen2",
                   "dtype":"torch.bfloat16","trainable_parameters":5046272,"device":"cuda:0"}}
    write_json(directory/"run_config.json",config)
    ids = [f"example:{i}" for i in range(1000)]
    mixture = tmp_path/"mixture.jsonl"
    mixture.write_text("".join(json.dumps({"id":i})+"\n" for i in ids))
    events = []
    for step in range(1,127):
        offset = (step-1)%63*16
        events.append({"step":step,"epoch":(step-1)//63+1,"example_ids":ids[offset:offset+16],
                       "mean_loss":2.,"lr_after_update":4e-5*(1-step/126)})
    (directory/"step_example_ids.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
    for step in [*range(10,126,10),126]:
        cp = directory/f"checkpoint-{step}"
        cp.mkdir()
        write_json(cp/"adapter_config.json",{"r":16,"lora_alpha":64,"target_modules":["q_proj","v_proj"]})
        (cp/"adapter_model.safetensors").write_bytes(b"fixture")
    return directory,mixture,events


def test_126_steps_and_partial_epoch_group(tmp_path):
    directory,mixture,events = make_run(tmp_path)
    assert len(events[62]["example_ids"]) == len(events[125]["example_ids"]) == 8
    assert validate_run(directory,mixture,42)["optimizer_steps"] == 126


@pytest.mark.parametrize("error",["shuffle","steps","nan","lr","missing_weights","wrong_model","wrong_budget"])
def test_reject_corrupt_training(tmp_path,error):
    directory,mixture,events = make_run(tmp_path)
    if error == "shuffle":
        events[0]["example_ids"].reverse()
    elif error == "steps":
        events.pop()
    elif error == "nan":
        events[10]["mean_loss"] = float("nan")
    elif error == "lr":
        events[5]["lr_after_update"] = .1
    elif error == "missing_weights":
        (directory/"checkpoint-126/adapter_model.safetensors").unlink()
    elif error in {"wrong_model","wrong_budget"}:
        config = json.loads((directory/"run_config.json").read_text())
        if error == "wrong_model":
            config["runtime"]["hidden_size"] = 1024
        else:
            config["epochs"] = 3
        write_json(directory/"run_config.json",config)
    (directory/"step_example_ids.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
    with pytest.raises(ValueError):
        validate_run(directory,mixture,42)


def test_reject_incomplete_or_nonfinite_measurements(tmp_path):
    traits = [str(i) for i in range(7)]
    rows = [{"step":step, **{f:{t:0. for t in traits} for f in ("scores","drift","normalized_drift_percent")}}
            for step in [0,*range(10,126,10),126]]
    path = tmp_path/"trajectory.json"
    write_json(path,rows)
    assert validate_trajectory(path,traits)["all_finite"]
    rows[-1]["normalized_drift_percent"]["0"] = float("nan")
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        validate_trajectory(path,traits)
    write_json(path,rows[:-1])
    with pytest.raises(ValueError):
        validate_trajectory(path,traits)
