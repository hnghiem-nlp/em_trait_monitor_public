# Trait-space Monitoring for Emergent Misalignment During Supervised Finetuning

Reference implementation for the paper
[Trait-space Monitoring for Emergent Misalignment During Supervised Finetuning](https://arxiv.org/abs/2606.07631) (arXiv:2606.07631).

Code for the two-phase trait-space monitoring pipeline:

1. **Phase 1 — Trait-direction extraction.** Once per base model, extract
   seven alignment-relevant trait directions in activation space using
   contrastive system prompts
   (mean-difference of contrastive activations, Eq. 1 in the paper).
2. **Phase 2 — Per-checkpoint measurement.** For each LoRA finetuning run,
   project the model's hidden states at every checkpoint onto the seven
   trait directions to obtain a 7-dimensional drift trajectory.

This package contains only the code paths needed to reproduce these two
phases. Post-hoc analyses (PC1, regressor selection, alarm calibration,
figure rendering) use standard libraries and can be implemented using the produced artifacts. 

## Requirements

- Python 3.9+ (type-union annotations are deferred via
  `from __future__ import annotations`).
- A CUDA GPU is required for both phases. Phase 1 with full layer selection
  runs on the order of an hour per model on a single A6000-class GPU
  (dominated by steering-based layer selection over the candidate set);
  pass `--layer <l>` or `--skip-layer-selection` to skip it.


```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
``` 

## Setup

### 1. Configure model paths

`configs/model_paths.yaml` maps short keys (e.g. `llama3-8b`) to either
HuggingFace IDs or local filesystem paths. Default values are HF IDs;
the loader passes them to `transformers.AutoModel.from_pretrained()`,
which will use your local HuggingFace cache (set `HF_HOME` to control
its location).

If you have local weight copies, replace the values with absolute paths.

### 2. Prepare finetuning datasets

The seven perturbation datasets used in the paper (4 calibration + 3
held-out) are drawn from public sources (the Betley emergent-misalignment
repo, HuggingFace). See **`data/README.md`** for the source of each dataset
and the required file format. Each dataset must be saved as
`data/<pert>_prompts.json` before running Phase 2, one row per training
example in `{"messages": [{"role": "...", "content": "..."}, ...]}` format.

## Running the pipeline

### Phase 1: extract trait directions for a model (once per model)

```bash
python -m experiments.extract_directions --model llama3-8b
```

Outputs (per model):
```
results/directions/<model>/
    trait_directions.pt        # 7 unit-norm trait directions at l*
    probe_results.json        # per-trait linear probe accuracy
    layer_selection.json      # chosen target layer l*
```

Repeat for each base model you intend to finetune
(`mistral-7b`, `qwen25-7b`, `gemma2-9b`, etc.).

### Phase 2: train + measure a single (model, pert, seed) cell

```bash
python -m experiments.train_and_measure \
    --model llama3-8b \
    --data-source bad_medical \
    --seed 42 \
    --lr 4e-5
```

Or via the convenience driver (which auto-runs Phase 1 if needed):

```bash
bash scripts/run_pipeline.sh llama3-8b bad_medical 42
```

Defaults: `lr=4e-5`, LoRA rank 16, alpha 64, dropout 0.05, target modules
`q_proj, v_proj`, 2 epochs, checkpoint saves every 10 steps. These can be
overridden via CLI flags; see
`python -m experiments.train_and_measure --help`.

Note on checkpoint counts: `trajectory.json` contains the step-0 baseline,
one entry per periodic save (steps 10, 20, ..., 120 at the default
configuration), and one entry for the final adapter. Adjust for checkpoint granularity per your preference.

Outputs (per cell, under `seed_{seed}/`):
```
trajectory.json               # 7D trait projections at every checkpoint
activations.pt                # raw activations at the target layer
checkpoints/checkpoint-{step}/
                              # per-step LoRA weights (HuggingFace format)
lora_adapter/                 # final-step adapter
run_manifest.json             # (model, pert, lr, seed, config hashes)
train_and_measure.log
```

`trajectory.json` is the primary artifact: a JSON list with one entry per
checkpoint, each containing the seven trait projections both as raw signed
scalars (`projections`, in the model's native activation-space units) and
cosine-normalized (`projections_normalized`, divided by the step-0
`step0_activation_norm`; these are the units the paper reports).

## From trajectories to the paper's analyses

This release produces the raw material for the paper's analyses; the
analysis scripts themselves are intentionally excluded. To go from the
Phase 2 outputs to the paper's results:

1. **EM labels.** For each saved checkpoint (`checkpoints/checkpoint-{step}`,
   HuggingFace format), generate one response per Betley prompt
   (72 prompts, temperature 1.0, max 600 new tokens) and grade them with
   `python -m src.evaluation.betley_judge` (two-pass aligned + coherent;
   see `data/README.md` for the required Betley YAML files and SECRETS
   setup). A checkpoint's EM rate is the misaligned fraction among
   scoreable responses; the paper labels checkpoints dangerous at
   EM > 5%. Note the judge incurs OpenAI API cost (~144 calls per
   checkpoint).

2. **Geometry (paper Sec. 4.2).** Compute per-checkpoint drift from
   `trajectory.json` as the `projections_normalized` values minus the
   step-0 entry, and run PCA over final-checkpoint drift vectors pooled
   across runs. `projections_normalized` is already cosine-normalized: each
   raw projection is divided by `step0_activation_norm` (the mean per-prompt
   activation L2 norm at step 0, i.e. mean ||h||), the single scalar that
   makes drift comparable across models. The raw `projections` and the
   `step0_activation_norm` scalar are also stored if you need to
   re-normalize from `activations.pt`.

3. **Detector (paper Sec. 4.3).** Fit a per-model regressor (the paper's
   headline uses scikit-learn RandomForestRegressor) from the 7D drift
   vectors to continuous EM rate on calibration runs; flag checkpoints
   when predicted EM exceeds 5%. Protocol details (LOPO/LOSO
   cross-validation, threshold selection) are specified in the paper's
   appendices.

Expect qualitative reproduction (PC1 dominance, dangerous-vs-benign
magnitude separation, detection metrics within the paper's reported
confidence intervals) rather than bit-exact numbers: GPU nondeterminism,
hardware/dtype differences, judge-API variation, and 72-prompt binomial
noise all move per-checkpoint values.

## Configuration files

| File | Contents |
|---|---|
| `configs/models.yaml` | Per-model architecture metadata (num layers, hidden dim, candidate layers for `l*`, dtype, LoRA defaults) |
| `configs/model_paths.yaml` | HF IDs or local paths for each model key |
| `configs/traits.yaml` | The seven trait names + per-trait positive/negative system prompts and the user-question pool |
| `configs/contrastive_passages.yaml` | Base-model contrastive passage pairs (used when extracting from a base/non-instruct model) |

## Layout

```
src/                          # core library, no scripts
    extraction/               # contrastive trait-direction extraction (Phase 1)
    measurement/              # checkpoint hidden-state projection (Phase 2)
    evaluation/               # linear probes + Betley EM judge (and appendix judges)
    utils/                    # config + path helpers + logging
experiments/
    extract_directions.py     # Phase 1 driver
    train_and_measure.py      # Phase 2 driver
configs/                      # YAML configs (see above)
data/                         # dataset docs + expected files (see data/README.md)
scripts/run_pipeline.sh       # generic single-cell launcher
```


## Citation

If you use this code or reference the paper, please cite:

​```bibtex
@article{nghiem2026trait,
  title={Trait-space Monitoring for Emergent Misalignment During Supervised Finetuning},
  author={Nghiem, Huy and Ho, Sy-Tuyen and Wiegreffe, Sarah and Daum{\'e} III, Hal},
  journal={arXiv preprint arXiv:2606.07631},
  year={2026}
}
​```

## License

MIT (see `LICENSE`).
