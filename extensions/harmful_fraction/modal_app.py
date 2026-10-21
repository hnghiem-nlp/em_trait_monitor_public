"""Run handoff 2 on a single persistent Modal A100; no judge API keys required."""
from pathlib import Path
import modal

REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/repo")
app = modal.App("cs2881-em-harmful-fraction")
cache = modal.Volume.from_name("cs2881-hw0-model-cache", create_if_missing=True)
results = modal.Volume.from_name("cs2881-em-harmful-fraction", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.7.0", "transformers==4.56.1", "accelerate==1.8.1",
                    "peft==0.15.2", "datasets==3.6.0", "numpy==2.2.6",
                    "pyyaml==6.0.2", "tqdm==4.67.1", "matplotlib==3.10.3")
    .env({"HF_HOME":"/cache/huggingface", "TOKENIZERS_PARALLELISM":"false",
          "MPLBACKEND":"Agg", "PYTHONUNBUFFERED":"1"})
)
if modal.is_local():
    image = (image
        .add_local_dir(REPO/"scratch_reproduction", "/repo/scratch_reproduction",
                       ignore=["data/", "__pycache__/", "*.pt", "*.log", "**/checkpoint-*/"])
        .add_local_dir(REPO/"configs", "/repo/configs")
        .add_local_dir(REPO/"extensions/harmful_fraction", "/repo/extensions/harmful_fraction",
                       ignore=["artifacts/", "deliverables/", "__pycache__/"])
    )


@app.function(image=image, gpu="A100-80GB", max_containers=1, timeout=24*60*60,
              volumes={"/cache":cache, "/results":results})
def sweep(training_seed: int = 42, instrument_dir: str = "", fractions: str = ""):
    import sys
    sys.path.insert(0, "/repo/extensions/harmful_fraction")
    from pipeline import run_pipeline
    return run_pipeline(results.commit, cache.commit, training_seed, instrument_dir, fractions)


@app.function(image=image, gpu="A100-80GB", max_containers=1, timeout=24*60*60,
              volumes={"/cache":cache, "/results":results})
def five_sweeps():
    """Complete exactly the 25 missing conditions in the approved 35-run design."""
    import sys
    import json
    sys.path.insert(0, "/repo/extensions/harmful_fraction")
    from pipeline import run_pipeline
    jobs = [(42, 1, "75"), (42, 2, ""), (42, 3, ""),
            (123, 1, "0,5,10,25,75"), (789, 1, "0,5,10,25,75")]
    # Existing runs are reused, not re-labeled or retrained.
    expected = {
        "/results/fraction_v2_af02f09860eb_data42_train42": 6,
        "/results/fraction_v2_270ff51deae3_data42_train123_p50-100": 2,
        "/results/fraction_v2_5c3526344cba_data42_train789_p50-100": 2,
    }
    for root, count in expected.items():
        status = json.loads((Path(root)/"status.json").read_text())
        assert status["stage"] == "complete" and len(status["completed"]) == count
    completed = []
    for seed, repeat_id, fractions in jobs:
        completed.append(run_pipeline(results.commit, cache.commit, seed, "", fractions, repeat_id))
        (Path("/results")/"five_sweeps_progress.json").write_text(json.dumps(completed, indent=2))
        results.commit()
    return completed


@app.local_entrypoint()
def main(training_seed: int = 42, instrument_dir: str = "", fractions: str = "", approved_five_sweeps: bool = False):
    if approved_five_sweeps:
        print(five_sweeps.remote())
    else:
        print(sweep.remote(training_seed, instrument_dir, fractions))
