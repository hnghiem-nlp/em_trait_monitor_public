# Authorized completion of the 35-condition design

On October 18, 2026, the user authorized completing every missing approved condition.

Fresh audit: 27 conditions passed training, order, checkpoint, provenance, activation-grid and baseline checks. Seed 123 at 5% passed training checks but had no activation trajectory. Seven conditions had no saved run: seed 123 at 10/75%; seed 789 at 0/5/10/25/75%.

`recover_35.py` uses the frozen original code snapshot `dd16d4da04f5`, original pinned model revision, mixture bytes, sequential order, monitor tensors and training configuration. It measures the trained 5% model without retraining it, then trains only absent conditions. Existing partial/invalid training causes an explicit failure, not silent overwrite. The final audit requires exactly 35 verified conditions and checks that the 27 previously verified final adapter hashes are unchanged.

Run the audit only:

```sh
MODAL_PROFILE=penrose modal run --env vlln-test extensions/harmful_fraction/recover_35.py
```

Run the authorized recovery (first check for an active recovery app to avoid concurrent jobs):

```sh
MODAL_PROFILE=penrose modal run --detach --env vlln-test extensions/harmful_fraction/recover_35.py --run
```

After it completes:

```sh
MODAL_PROFILE=penrose modal run --env vlln-test extensions/harmful_fraction/export_35.py
MODAL_PROFILE=penrose MODAL_ENVIRONMENT=vlln-test python extensions/harmful_fraction/live_results.py
```

`finish_35.py` is this session's finite continuation, waiting on the active job and then running those finalization commands. It does not create a recurring automation.

Evidence downloads go to `deliverables/recovery35/`. Exact seven-fraction mixture files are downloaded once. Each condition includes provenance, protocol, configuration, step-example IDs, training/measurement verification and all measured trait checkpoints. Model weights stay on the persistent `cs2881-em-harmful-fraction` volume; paths and final adapter hashes are in the final audit. A successful launch or initial audit is not evidence of 35/35 completion: require `final_audit.json` with `verified: 35` and `completion_status.json` with `stage: complete`.

The original 28-model output-evaluation manifests remain frozen. New training models must be evaluated under a new or explicitly extended manifest; training completion does not imply behavioral judging completion. The prior expanded evaluation also retains five invalid Claude judgments pending adjudication; its automated-complete flag remains false.
