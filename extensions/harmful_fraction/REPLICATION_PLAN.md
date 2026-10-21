# Training-seed replication plan

Recorded before inspecting seed-123 endpoint results. Seed-42 was the exploratory
sweep that identified the 50%-versus-100% pattern.

## Question

Is final PC1 magnitude larger at 50% harmful data than at 100%, when the
training randomness changes but data, example order, model, monitor and budget
stay fixed?

## Outcomes and reporting

- Primary outcome: final absolute PC1 at optimizer step 126.
- Primary comparison: the paired difference, 50% minus 100%, within a seed.
- Report every seed, including negative differences, without excluding runs
  based on their outcome. Technical failures may be rerun with the same seed
  and protocol, with the failure recorded.
- Show the individual pairs and their mean. Uncertainty must use independent
  training runs, not the 115 evaluation prompts or saved checkpoints as if they
  were independent model replications.
- Treat seed-42 as discovery evidence. Clearly label the new seeds as follow-up
  replications rather than presenting the whole exercise as preregistered.
- Five seeds improve precision but do not guarantee significance. Avoid a
  significance claim based on a favorable choice of test. Any paired t interval
  needs its distributional assumptions stated; report the raw differences too.
- Training-seed replication does not test uncertainty over training-data samples
  or alternative data orders, and it does not establish behavioral harm.

## Seeds fixed in advance for a possible expansion

42 (already observed), 123 (currently running), 789, 2026, 31415.
Do not replace these after seeing their results.

The user is choosing whether to apply the five seeds to only 50% and 100%
(10 full runs total) or all six fractions (30 full runs total). Only the two
seed-123 conditions were launched so far. No additional seeds are authorized
until that scope is resolved.

## Existing-data analyses

Onset at reference levels 8 and 15, extra final drift over the 0% condition,
and an arithmetic decomposition of PC1's 100%-minus-50% change by trait. These
are exploratory analyses; reference levels do not establish behavioral safety.

## Completed seed-123 follow-up

Both runs completed successfully. Protocol, data/order hashes, monitor hashes, core training/measurement code, and 126-step budgets match the original sweep. Final |PC1|: 50% = 26.606871; 100% = 22.869580; paired difference = +3.737291. Original seed42 difference = +2.834124. Both seeds show the same ranking; two seeds are insufficient for a reliable significance claim. Further runs await scope selection.

## Superseding user-approved scope, October 16

The user chose five sweeps with seeds [42,42,42,123,789], each across
[0,5,10,25,50,75,100]. This is 35 training runs, only three distinct training
seeds. Seed42 repeats are reproducibility checks and must not increase the
independent-seed sample size or give seed42 extra weight in error bars.
Reuse original seed42 six conditions and completed 123/789 pairs (10 runs).
Launch exactly 25 missing runs: seed42 repeat1 at75; repeat2 and repeat3 all
seven; seed123 and seed789 at0,5,10,25,75. No more seed2026 or31415 training.
The earlier seed2026 pair did complete before this scope change; retain it
as supplemental evidence outside the chosen design, not silently discard it.
75% uses the same nested source pools and fixed shuffle as existing fractions.
Original trainer, model, monitor, data seed and order remain unchanged.
Estimated additional GPU runtime 60–90 minutes; budget $3–$5 with CPU/RAM
and overhead, before credits. Actual billed costs may differ.
