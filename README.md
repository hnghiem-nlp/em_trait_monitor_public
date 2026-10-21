# Trait-space checkpoint monitoring

Parameterized code for contrastive direction extraction, semantic steering
validation, and checkpoint measurement. Paper-specific settings belong in the
reproduction notebooks and configuration files, rather than script-level rules.

The [six reviewer notebooks](notebooks/README.md) include saved inline tables
and figures. Rerunning them requires the separate CPU-readable supplement;
analysis inputs are not stored in this repository.

## Output judging and harmful-fraction experiments

See [the output-judge guide](extensions/output_judge/README.md) for the Claude and Prometheus judge code, saved answers and judgments, seven-trait activation comparisons, plots, and reproduction instructions. The completed evaluation covers 27 available final trained models plus the untuned base; intermediate checkpoints are not included.

## Installation

Python 3.11 and a CUDA GPU are recommended. Tested versions are recorded in
`constraints.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt
```

Configure models in `configs/models.yaml`. Gated models may require HuggingFace
access approval. Optional local-weight overrides belong in
`configs/model_paths.yaml`. Do not commit private paths or credentials.

## Extract directions

```bash
python -m experiments.extract_directions --model llama3-8b --layer 14
```

Directions are unit-normalized differences between positive and negative
activation means, collected at the final input token with an assistant-generation
prompt. Gemma represents system instructions as a preceding user/assistant
exchange. The supplied prompt configuration contains 30 questions and five
system prompts per polarity, but these counts are not enforced.

Use `--prompt-config` for another YAML with `traits` and `questions`,
`--traits` for specific trait names, `--revision` to pin model weights, and
`--output-dir` to choose a destination. Without `--layer`, the model
configuration supplies the default. Optional `--probe-diagnostics` adds
linear-probe separation scores, not semantic validation.

Outputs include `trait_directions.pt`, `per_prompt_acts.pt`,
`layer_selection.json`, and `extraction_manifest.json`. They are written
together after extraction completes. Existing directories are not overwritten.

To independently check hashes, layer metadata, and cache lineage:

```bash
python scripts/audit_extraction_bundle.py --model llama3-8b
```

## Validate steering

Obtain the Betley rubric files as described in [data/README.md](data/README.md)
and set `OPENAI_API_KEY`, or use a gitignored `SECRETS` file.

```bash
python -m experiments.causal_steering.run_semantic_steering_v2 plan \
  --model llama3-8b --layers 14 16 --prompts-per-trait 20
# Repeat with "generate", then "judge", using the same arguments.
```

Validation uses blinded trait-specific comparisons and a separate coherence
judge. A common steering dose must pass the coherence gate at every candidate
layer. Semantic scores weight traits equally and include paired prompt-bootstrap
intervals. Validation records its argmax but does not change extraction settings.

The supplied YAML contains the paper's four-model protocol. Use `--config`,
`--layers`, `--prompts-per-trait`, and `--output-dir` for another experiment.
`--run-kind pilot` supplies smaller defaults without restricting the model or
valid candidate layers. Generation and judging resume independently. A changed
run configuration requires a new output directory to avoid mixing records.
Generation needs a GPU, and judging incurs API cost.

The default full-run results can be checked with
`python scripts/audit_semantic_steering_v2_full.py --model <model>`.
The audit recomputes scores and intervals from saved judgments.

## Train and measure

Prepare a [messages-format dataset](data/README.md), then run:

```bash
python -m experiments.train_and_measure \
  --model llama3-8b --data-source bad_medical --seed 42 \
  --n-samples 1000 --lr 4e-5 --lora-rank 16 --lora-alpha 64
```

Any dataset name resolving to `data/<name>_prompts.json` is accepted.
Omitting `--n-samples` uses the full file. When supplied, the run seed controls
sampling. Layers and trait names come from the supplied directions, not a fixed
seven-trait requirement.

Use `--directions-dir` for another direction set, `--layer` if its layer
metadata is absent, `--prompt-config` for evaluation prompt groups, and
`--output-dir` for another destination. Training options include learning rate,
epochs, rank, alpha, checkpoint interval, and full finetuning. Use `--help`
for the complete interface.

Outputs include a metadata-wrapped `trajectory.json`, prompt-level
`activations.pt`, a training `run_manifest.json`, checkpoints, and a final
adapter or model. Projections are stored both raw and normalized by the mean
per-prompt activation norm at step zero. The final adapter can alias the last
numeric checkpoint, so analyses should avoid counting both independently.

`--train-only` and `--measure-only` separate the stages. Measurement uses
stored training metadata when available and records its own direction and prompt
hashes. Direction validity and available hashes are checked automatically.
Source extraction caches are required only for a separate lineage audit, not
for ordinary measurement.

## Reproduction scope and tests

The paper specifies the settings for reported results; the notebooks record
those parameters. The separate reviewer supplement contains checkpoint mean
activations, trajectories, behavioral labels, baseline weights/features, and
reference tables. Copy its `results/` and `review_artifacts/` directories into
this checkout before rerunning the notebooks. See
[notebooks/README.md](notebooks/README.md) for the distinction between locally
recomputed results and saved upstream analyses. Prepared finetuning datasets,
model checkpoints, and live API/GPU experiments are not included.

```bash
pip install -r requirements-dev.txt -c constraints.txt
python -m pytest -q
python scripts/audit_semantic_steering_v2_prompts.py
```

Tests use synthetic activations and mocked rubrics, without credentials or
model weights.

## License

MIT. Publication links and citation metadata are omitted from this review copy.
