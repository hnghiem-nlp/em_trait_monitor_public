# Checkpoint 2 behavioral evaluation

Inference and evaluation only. No adapter is trained or modified.

## Frozen scope

`build_manifest.py` joins the saved Modal adapter inventory to measured activation steps, restricts training seeds to 42/123/789, and preserves repeat IDs. By default it selects final step 126. `--all-steps` includes intermediate measured steps. Base model is evaluated once. Missing trajectories are explicitly excluded. The manifest digest namespaces all outputs.

133 prompts: the exact 115 original monitor prompts plus 18 author-written development probes. These are not a held-out validated benchmark. Use the original set as the primary comparison; targeted probes are diagnostic and reported separately.

## Run

From repository root, using the existing HW virtual environment:

```sh
MODAL_PROFILE=penrose modal run --env vlln-test extensions/output_judge/checkpoint2/inventory.py
python extensions/output_judge/checkpoint2/build_manifest.py
MODAL_PROFILE=penrose modal run --env vlln-test extensions/output_judge/checkpoint2/generate.py
MODAL_PROFILE=harsh-24539 modal run --env main extensions/output_judge/checkpoint2/judge.py --pilot
MODAL_PROFILE=harsh-24539 modal run --env main extensions/output_judge/checkpoint2/judge.py
python extensions/output_judge/checkpoint2/analyze.py
python extensions/output_judge/checkpoint2/category_analysis.py
python extensions/output_judge/checkpoint2/inspect_answers.py
python extensions/output_judge/checkpoint2/repeat_analysis.py
python extensions/output_judge/checkpoint2/plots.py
python extensions/output_judge/checkpoint2/finish_report.py
```

Generation uses pinned Qwen revision and vLLM 0.15.0, deterministic greedy decoding and max 768 output tokens; every answer records finish reason. Truncation is reported, not silently treated as model incompetence. This runtime differs from the earlier Transformers pilot, so do not combine the two batches as identical repeated generations.

Claude Sonnet 5 is blind to model identity, fraction, seed and activation values. It rates overall quality, coherence, and seven traits conditional on observability. Confidence measures expressed certainty, not calibration. Identical prompt/answer/truncation/protocol combinations share a cached judgment. All raw attempts and failures are retained. Evidence IDs are resolved to supplied answer lines; a leading judge preamble may be parsed separately without changing final JSON values. Credentials remain in HW1's existing Modal secret.

Structural validation checks metric coverage, score ranges and source-line evidence references, and flags grossly inconsistent overall/component ratings. It cannot establish judge correctness. Audit factual claims, low scores, unexpected changes and a sample of high ratings before drawing conclusions. Never silently overwrite judge scores with an analyst's preferred score.

## Analysis

Compare each model with base on the same assessable prompts. Independent-seed summaries use repeat 1 only. Repeat 2/3 stay in per-model tables as reproducibility checks. Between-seed SD is not a confidence interval; prompt count is not training-seed replication. Do not pool repeated checkpoints as independent observations in significance tests. Missing seeds are reported, not imputed. Training mixture still confounds harmfulness with medical versus numerical domain.

The original activation monitor measures final prompt-token hidden states at layer 16 before an answer is generated. Behavioral evaluation judges generated text. Their scales differ. Correlations across conditions are exploratory, not causal or a validated safety detector.

## Recreate the comparison figures from saved results

After installing `extensions/output_judge/requirements-analysis.txt`, run from the repository root:

```sh
python extensions/output_judge/checkpoint2/plot_original_run.py
python extensions/output_judge/checkpoint2/plot_metric_comparison.py
```

These commands read the committed CSV results and do not call a model, launch training, or incur API charges. The first plots seed 42, repeat 1. The second compares activation and answer scores for all seven traits, with separate scales, seed counts and sample SD bars. The saved report is `runs/f098104a25b14102/report.html` relative to this directory.
