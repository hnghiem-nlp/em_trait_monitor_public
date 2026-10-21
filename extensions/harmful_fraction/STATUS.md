# Audit and run status — 2026-10-16

Completed corrected run:
https://modal.com/apps/penrose/vlln-test/ap-5wgVWa011An7SEuNmFdBp5

Authoritative live status: volume `cs2881-em-harmful-fraction`,
`/latest_status.json`. The corrected run completed all six conditions and analysis on October 16,
2026, at approximately 01:29 EDT. The full pipeline, including smoke testing,
training and measurement, took 13 minutes 23 seconds. Results are preserved
under `/fraction_v2_af02f09860eb_data42_train42` on the result volume.

Local deliverables: `deliverables/output/handoff_2_results.pptx` (two editable
slides) and `deliverables/run_data/` (raw trajectories, provenance, validation,
metrics and six figures). Local recomputation of all final/peak PC1 and 7D
summary values matched the downloaded metrics. Every condition had 126
optimizer steps, the complete 14-point measurement grid, one common monitor,
and zero relative base-activation error. Saved adapters remain on Modal.

Final PC1 magnitudes at 0/5/10/25/50/100% were
12.729 / 19.435 / 23.438 / 27.169 / 28.868 / 26.034. The response is not monotone.
Piecewise leave-one-fraction-out RMSE was 4.849 versus 9.399 for a linear fit.
These are one-seed representation results, not behavioral safety scores.

## GitHub scope checked

Fetched origin and checked all three published branches plus open/closed PRs.
There are no open PRs. The only closed PR is #1, handoff 1. The earlier
`research/mixture-path-dependence` branch has a mixture builder and plan, but
no completed six-fraction sweep. Handoff 3 has no published implementation in
these branches. Unpushed or private teammate work cannot be ruled out.

## Problems found and fixed

- Initial code used seed 43 for the benign pool. Corrected to seed 42 for both
  sources, matching the reproduction and handoff 1.
- Initial runner reshuffled each epoch. Corrected to the handoff-1 fixed random
  convention for every fraction. This makes 25% match its exact training order.
- Initial reruns could reuse a stale persisted source workspace and overwrite
  provenance labels. Source snapshots/results are now keyed by code hash and
  existing outputs require identical provenance.
- Initial tests covered synthetic mixture/fit behavior, but not compatibility
  with the teammate's actual data, training order, or runtime configuration.
  Added those checks and fail-closed runtime validation.
- Original monitor tensors are still unavailable. Rebuilt-monitor identity with
  older runs remains unverified and is explicitly reported.

## Evidence

The downloaded datasets have 7,049 bad_medical and 14,926 evil_numbers examples.
All 1,000 selected-example hashes match the committed handoff-1 manifest.
The corrected 25% file matches its random order hash and all 2,000 example
presentations in the teammate's two-epoch training log.

The initial two-step real-Qwen smoke test produced finite measurements for all
seven traits and reported 5,046,272 trainable LoRA parameters. The corrected run
repeats the smoke test with the audited ordering and adds fresh base-model
verification. Fourteen local tests pass.

## Preserved previous attempts

- `ap-9owZiKfvJPMyyTaOU66dfk`: stopped after module-path startup failure.
- `ap-tD2AYBIdteUT9NF7q7cUjS`: stopped during the audit after the original
  smoke test. Its outputs under `/sweep_seed42` are superseded. Its audited
  source downloads and rebuilt monitor remain inputs to v2.

Exact comparability with older numerical trajectories requires the group's
original `.pt` instrument files (or remeasurement of older adapters using our
shared monitor). Checkpoints from the new sweep are retained for remeasurement.
