"""Durable handoff-2 pipeline with immutable code/data/monitor provenance."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from make_fraction_mixtures import build, read_source
from verify import (sha256, write_json, verify_mixtures, validate_run, validate_trajectory)

REPRO_FILES = ["03_build_trait_directions.py", "04_measure_base_position.py",
               "09_finetune_dataset.py", "10_measure_dataset_drift.py"]
TRAITS = ["honesty","helpfulness","harmlessness","power_seeking","corrigibility","sycophancy","confidence"]


def selected_fractions(value=""):
    allowed = (25, 0, 5, 10, 50, 75, 100)
    if not value:
        return allowed
    requested = tuple(int(item.strip()) for item in value.split(","))
    if len(set(requested)) != len(requested) or any(p not in allowed for p in requested):
        raise ValueError("Fractions must be unique members of 0,5,10,25,50,75,100")
    return tuple(p for p in allowed if p in requested)


def run_pipeline(commit, cache_commit, training_seed=42, instrument_dir="", fractions="", repeat_id=1):
    if repeat_id not in (1, 2, 3):
        raise ValueError("repeat_id must be 1, 2, or 3")
    selected = selected_fractions(fractions)
    from huggingface_hub import snapshot_download
    from analyze import analyze
    source_repo = Path("/repo")
    extension = source_repo/"extensions/harmful_fraction"
    spec = json.loads((extension/"protocol_spec.json").read_text())
    code_files = [source_repo/"scratch_reproduction"/p for p in REPRO_FILES]
    code_files += sorted(extension.glob("*.py")) + [extension/"protocol_spec.json", source_repo/"configs/traits.yaml",
        source_repo/"scratch_reproduction/artifacts/qwen_seed42_pc1.json",
        source_repo/"scratch_reproduction/artifacts/ordered_mixture_seed42/manifest.json",
        source_repo/"scratch_reproduction/artifacts/mixture_random_seed42/step_example_ids.jsonl"]
    code_hashes = {str(p.relative_to(source_repo)):sha256(p) for p in code_files}
    import hashlib
    code_id = hashlib.sha256(json.dumps(code_hashes, sort_keys=True).encode()).hexdigest()[:12]
    # Different source code always gets a fresh workspace and result namespace.
    workspace = Path("/results/code")/code_id
    if not workspace.exists():
        shutil.copytree(source_repo, workspace)
    for relative, expected in code_hashes.items():
        if sha256(workspace/relative) != expected:
            raise ValueError("Persisted source snapshot changed")
    scripts = workspace/"scratch_reproduction"
    suffix = "" if len(selected) == 7 else "_p" + "-".join(map(str, selected))
    suffix += f"_repeat{repeat_id}"
    root = Path(f"/results/fraction_v2_{code_id}_data42_train{training_seed}{suffix}")
    root.mkdir(parents=True, exist_ok=True)
    state = {"schema_version":2, "data_seed":42, "training_seed":training_seed,
             "root":str(root), "code_id":code_id, "stage":"starting", "completed":[],
             "started_unix":time.time(), "selected_fractions":list(selected)}
    print("RUN ROOT:",root,flush=True)

    def save(stage):
        state.update(stage=stage,updated_unix=time.time())
        write_json(root/"status.json",state)
        write_json(Path("/results/latest_status.json"),state)
        commit()

    def run(script, arguments=(), label=None):
        label = label or Path(script).stem
        log_path = root/f"{label}.log"
        print(f"Starting {label}; log: {log_path}",flush=True)
        with log_path.open("a") as log:
            process = subprocess.Popen([sys.executable,str(script),*map(str,arguments)],
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,cwd=workspace)
            for line in process.stdout:
                print(line,end="",flush=True)
                log.write(line); log.flush()
            code = process.wait()
        commit()
        if code:
            raise RuntimeError(f"{label} failed with exit {code}; see {log_path}")

    try:
        save("verifying_inputs")
        # Reuse the audited, immutable dataset bytes from the initial download.
        # These were independently checked against all 1,000 team content hashes.
        sources = root/"sources"
        sources.mkdir(exist_ok=True)
        for name, metadata in spec["sources"].items():
            path = sources/f"{name}.jsonl"
            original = Path("/results/sweep_seed42/sources")/path.name
            if not path.exists():
                if not original.exists():
                    raise FileNotFoundError(f"Upload the audited source snapshot: {original}")
                shutil.copyfile(original,path)
            if sha256(path) != metadata["sha256"] or len(read_source(path)) != metadata["count"]:
                raise ValueError(f"Audited source snapshot mismatch: {name}")
        if not (root/"mixtures/manifest.json").exists():
            build(read_source(sources/"bad_medical.jsonl"),read_source(sources/"evil_numbers.jsonl"),
                  root/"mixtures",seed=42)
        mixture_check = verify_mixtures(root/"mixtures",scripts/"artifacts/ordered_mixture_seed42/manifest.json")
        write_json(root/"mixture_verification.json",mixture_check)
        manifest = json.loads((root/"mixtures/manifest.json").read_text())
        model_path = snapshot_download(spec["model_id"],revision=spec["model_revision"])
        os.environ["EM_MODEL_PATH"] = model_path
        # Preserve the already built monitor, or explicitly use a supplied original
        # bundle. Never silently rebuild/replace an instrument on resume.
        instrument = Path(instrument_dir) if instrument_dir else Path("/results/workspace/scratch_reproduction/artifacts")
        instrument_paths = [instrument/"trait_directions_qwen_layer16.pt", instrument/"base_position_qwen_layer16.pt",
                            scripts/"artifacts/qwen_seed42_pc1.json",workspace/"configs/traits.yaml"]
        hashes = {p.name:sha256(p) for p in instrument_paths}
        protocol = {"schema_version":2,"model":spec["model_id"],"model_revision":spec["model_revision"],
            "data_seed":42,"training_seed":training_seed,"repeat_id":repeat_id,"fractions":sorted(selected),
            "source_sha256":{k:v["sha256"] for k,v in spec["sources"].items()},
            "ordering":spec["ordering"],"comparison":spec["comparison"],
            "monitor_origin":"provided_bundle" if instrument_dir else "rebuilt_on_modal_initial_attempt",
            "monitor_identity_with_earlier_runs":"unverified; requires original tensors",
            "instrument_sha256":hashes,"layer_index":16,
            "normalization":"100 * drift / norm(mean base activation)",
            "code_sha256":code_hashes,"checkpoint_steps":[*range(10,126,10),126]}
        protocol_path = root/"protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Run protocol changed; refusing to mix outputs")
        write_json(protocol_path,protocol)
        cache_commit(); commit()
        protocol_hash = sha256(protocol_path)

        def train_measure(name,mix,max_steps=126,team_match=False):
            directory = root/name
            mixture = root/"mixtures"/mix["file"]
            provenance = {"protocol_sha256":protocol_hash,"instrument_sha256":hashes,
                "mixture_sha256":sha256(mixture),"model_revision":spec["model_revision"],
                "training_seed":training_seed,"data_seed":42}
            provenance_path = directory/"provenance.json"
            # No metadata is relabeled when resuming existing output.
            if directory.exists():
                if not provenance_path.exists() or json.loads(provenance_path.read_text()) != provenance:
                    raise ValueError(f"Missing or changed provenance for {name}")
            else:
                directory.mkdir()
                write_json(provenance_path,provenance)
                commit()
            completion = directory/"training_complete.json"
            if not completion.exists():
                if (directory/"run_config.json").exists() or list(directory.glob("checkpoint-*")):
                    raise RuntimeError(f"Interrupted partial training: {name}; inspect and restart explicitly")
                args = ["--dataset-jsonl",mixture,"--run-name",name,"--seed",training_seed,
                        "--output-root",root,"--save-every",10,"--sequential"]
                if max_steps != 126:
                    args += ["--max-steps",max_steps]
                save(f"training_{name}")
                run(scripts/"09_finetune_dataset.py",args,f"train_{name}")
            team_steps = scripts/"artifacts/mixture_random_seed42/step_example_ids.jsonl" if team_match else None
            training_check = validate_run(directory,mixture,training_seed,max_steps,team_steps)
            config = json.loads((directory/"run_config.json").read_text())
            if config["model"] != model_path:
                raise ValueError("Trainer did not use the pinned model snapshot")
            write_json(completion,training_check)
            commit()
            trajectory = directory/"trait_trajectory.json"
            if not trajectory.exists():
                save(f"measuring_{name}")
                run(scripts/"10_measure_dataset_drift.py",
                    ["--run-name",name,"--output-root",root,"--instrument-dir",instrument],f"measure_{name}")
            measurement_check = validate_trajectory(trajectory,TRAITS,max_steps)
            base_check = json.loads((directory/"base_validation.json").read_text())
            if base_check["relative_base_activation_error"] > .001:
                raise ValueError("Unadapted model differs from the baseline")
            write_json(directory/"verification.json",{"training":training_check,"measurement":measurement_check,
                                                        "baseline":base_check})
            commit()

        mix25 = next(m for m in manifest["mixtures"] if m["percent_harmful"] == 25)
        train_measure(f"smoke_seed{training_seed}",mix25,max_steps=2,team_match=True)
        state["smoke_passed"] = True
        # First full run is the exact data/order anchor to handoff 1.
        for p in selected:
            mix = next(m for m in manifest["mixtures"] if m["percent_harmful"] == p)
            name = f"mix_p{p:03d}_seed{training_seed}"
            train_measure(name,mix,team_match=(p==25))
            state["completed"].append(name)
            save(f"completed_{name}")
        save("analyzing")
        if set(selected) == {0,5,10,25,50,100}:
            analyze(root,scripts/"artifacts/qwen_seed42_pc1.json",training_seed)
        else:
            write_json(root/"subset_complete.json", {
                "fractions":list(selected), "training_seed":training_seed,
                "note":"Selected-condition replication; compare raw trajectories with the original sweep. No six-point fit performed."})
        import tarfile
        with tarfile.open(root/"report_bundle.tar.gz","w:gz") as archive:
            for path in root.rglob("*"):
                if path.is_file() and (path.parent.name=="analysis" or path.name in {
                    "trait_trajectory.json","provenance.json","run_config.json","protocol.json",
                    "verification.json","mixture_verification.json","base_validation.json","manifest.json","subset_complete.json"}):
                    archive.add(path,arcname=str(path.relative_to(root)))
        save("complete")
        return state
    except Exception:
        state["error"] = traceback.format_exc()
        save("failed")
        raise
