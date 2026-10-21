# Completed handoff 2 results

- `output/handoff_2_results.pptx`: two editable slides with speaker notes.
- `run_data/analysis/`: six figures, metrics, trajectories and exploratory fits.
- `run_data/mix_p*_seed42/`: raw measurements, configuration and verification.
- `run_data/protocol.json`: model, source, code and measuring-instrument hashes.
- `run_data/mixtures/manifest.json`: source IDs/content hashes and exact counts.
- `run_data/local_audit.json`: independent local summary checks.
- `instrument/`: exact rebuilt monitor used for these six runs. Tensor files are
  local only because the repository ignores `.pt`; durable copies remain on Modal.

All results come from Modal app `ap-5wgVWa011An7SEuNmFdBp5`, result volume
`cs2881-em-harmful-fraction`, root `/fraction_v2_af02f09860eb_data42_train42`.
Code snapshot: `/results/code/af02f09860eb`. All 78 trained checkpoint adapters,
training logs and per-step example IDs remain in the volume. The local bundle
omits adapter weights. Model measurements include step zero plus 13 checkpoints.

The monitor tensor copies are also preserved remotely under
`/workspace/scratch_reproduction/artifacts/`. Their hashes match the protocol.
These are our rebuilt files, not the missing originals from the team's prior runs.

## Interpretation

Final PC1 magnitude rises from 12.729 at 0% harmful data to 23.438 at 10%,
peaks at 28.868 for 50%, then falls to 26.034 at 100%. Peak-over-training values
show the same pattern. Units are percent of the base mean activation norm.

The aggregate drop at 100% does not apply to every trait: honesty drift becomes
more negative, while helpfulness and harmlessness drift become less negative.
A continuous one-hinge fit predicts held-out fractions better than a line in this
small sweep (RMSE 4.849 vs 9.399). This does not establish a phase transition.

For a budget of two additional runs, test 2% and 3% with the same protocol. For
four runs total, also repeat the observed 5% condition with training seeds 123
and 789, keeping the data fixed. The automatic analysis report additionally
suggests a longer conditional follow-up: repeat both new fractions with both
seeds if an early crossover persists. No follow-up runs have been launched.

One seed, no behavioral EM grading, and medical-vs-number domain differences
limit causal interpretation. Numerical pooling with older handoffs is unverified
until their original monitors become available. The retained adapters allow
remeasurement with those files without retraining.
