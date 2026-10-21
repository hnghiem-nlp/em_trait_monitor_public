# Harmful-data fraction / transition map (handoff 2)

Measure how representation drift changes with 0/5/10/25/50/100% bad_medical in
1,000-example mixtures of bad_medical and evil_numbers, holding training fixed.

## Audited protocol

- Qwen2.5-7B-Instruct, immutable revision in `protocol_spec.json`; bfloat16,
  28 layers, hidden size 3584. Measurement uses zero-based block index 16.
- LoRA rank 16, alpha 64, dropout .05, q_proj/v_proj: 5,046,272 trainable parameters.
- Full-sequence causal LM loss, sequence limit 512; AdamW LR 4e-5, weight decay
  .01, gradient clipping 1. Linear decay without warmup.
- Batch 4, accumulation 4; two epochs and 126 optimizer updates. The last
  accumulation group in each epoch contains eight examples, matching the
  reproduction. Save at 10,20,...,120,126.
- Data seed 42 for BOTH source pools, following the original trainer. Source
  pools contain 1,000 sampled indices each; every mixture takes nested prefixes.
- Shuffle each mixture once with seed 42001, then repeat that random order for
  both epochs (`--sequential`). This matches handoff 1's random-condition
  convention and still keeps every fraction randomly mixed.
- The 25% mixture's IDs, content hashes and order match the committed handoff-1
  manifest. Its exact optimizer-step order is checked against the teammate's
  actual training log, including the eight-example epoch-end groups.
- Frozen original `qwen_seed42_pc1.json`, never refit to the sweep outcomes.
- Preserve the reproduction's normalization: 100 × trait drift / norm(mean
  base activation), not mean(per-prompt activation norm).

The original 0% and 100% reproduction runs reshuffled each epoch. Their data
sets match our corrected source pools, but their presentation order differs.
We train all six conditions afresh using one ordering convention. Handoff 1
saved every five steps; our peaks use the handoff-2 ten-step grid, so old peaks
must be recomputed on the common grid before comparing peak values.

## Monitor comparability

Original direction/base .pt files were unavailable locally and were not found
in the connected Drive search. The first attempt rebuilt one monitor; v2 reuses
that saved monitor with unchanged hashes. Its base scores differ from the
handoff-1 baseline by up to 0.0491 raw units and from the original pure-run
baseline by up to 0.1795. This confirms the instruments are not proven identical.

All six new runs share one monitor and frozen PC1, so comparisons within this
sweep are consistent. Do not claim exact numerical comparability with earlier
experiments until the group's original tensors are supplied and the saved
adapters are remeasured. No retraining is needed just to change the measuring
instrument. `--instrument-dir` can select a supplied remote original bundle for
a new run; protocol guards prevent mixing instruments in an existing result.

Before measuring adapted checkpoints, the code disables the adapter and
remeasures the base. It rejects a relative base-activation discrepancy greater
than 0.001 and saves the actual discrepancy, instead of only inserting zeros
for step 0.

## Launch

From the repository root:

```sh
modal run --detach extensions/harmful_fraction/modal_app.py --training-seed 42
```

The configured Modal profile/environment is used. The function uses one
A100-80GB and at most one container. The v2 runner relies on the already audited
source snapshots under `/results/sweep_seed42/sources/` and the rebuilt tensors
under `/results/workspace/scratch_reproduction/artifacts/`, all on volume
`cs2881-em-harmful-fraction`. These are retained from the initial attempt.
A fresh deployment must upload those exact audited inputs first; it fails
rather than substituting another dataset or monitor. Model downloads use the
existing `cs2881-hw0-model-cache` volume.

Order of execution:

1. Verify exact source-file hashes, counts, mixture fractions, nesting, and
   agreement with the teammate's committed 25% data/order manifest.
2. Verify the pinned model snapshot and preserve an immutable source-code copy.
3. Train/measure a two-step real-7B smoke test and validate all outputs.
4. Run 25% first as the handoff-1 anchor, followed by 0/5/10/50/100%.
5. Generate metrics, six figures, exploratory fits and follow-up suggestions.

Results use a new namespace for every code version:
`/results/fraction_v2_<code-hash>_data42_train42`. Read `/latest_status.json` from
the result volume for the current root, stage, smoke result and completed runs.
The job continues after the local client disconnects.

## Verification and outputs

Each run verifies the actual model dimensions, dtype, trainable parameter count,
finite training loss, linear LR schedule, exact step sequence, example order,
checkpoint grid, adapter configuration/weights, and finite seven-trait results.
For 25%, it additionally checks every step against the teammate's real log.

Each fraction includes `run_config.json`, `step_example_ids.jsonl`,
`training_complete.json`, adapters, `trait_trajectory.json`, `provenance.json`,
`base_validation.json`, and `verification.json`. The root records source and
code hashes in `protocol.json`, plus `mixture_verification.json`.

`analysis/` contains final/peak PC1, signed PC1, final/peak 7D norm, trait
endpoints and per-trait peak magnitudes. Figures cover final PC1 vs fraction,
peak PC1 vs fraction, trait endpoints, PC1 trajectories, all seven trait
trajectories, and norm trajectories. `report_bundle.tar.gz` excludes model
weights and includes the mixture manifest and validation records.

Example status download:

```sh
modal volume get cs2881-em-harmful-fraction /latest_status.json latest_status.json
```

Use the reported root to download individual outputs or the report bundle.
Completed outputs may be reused only with identical provenance. Interrupted
partial training is rejected rather than overwritten; optimizer-state resume
is not implemented. All initial-attempt outputs are preserved separately and
must not be included in the corrected sweep.

## Interpretation and tests

Linear and continuous one-knot piecewise fits use least squares. Knots are
reselected inside each leave-one-fraction-out validation fold. Six points from
one seed support exploratory dose-response analysis, not a phase-transition
claim. Behavioral EM and capability evaluation are outside this first sweep.
The sources differ in topic as well as harmfulness, so domain mixing remains
part of the intervention. The original pure-run numbers are reference values,
not exact expected outputs of this sweep.

```sh
python -m pytest tests/test_fraction_mixtures.py tests/test_fraction_analysis.py tests/test_fraction_verification.py -q
```

Fourteen tests cover mixture integrity and original sampling/order, synthetic
linear and nonlinear fits, frozen-axis projection, incompatible instruments,
exact training-budget checks and rejection of corrupted runtime outputs. The
real public datasets have separately passed the committed teammate-manifest
and actual training-log comparison.
