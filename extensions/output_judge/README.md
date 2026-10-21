# Output judges and activation comparison

Start with [Checkpoint 2 instructions](checkpoint2/README.md). The complete saved evaluation is in [the results report](checkpoint2/runs/f098104a25b14102/report.html); download/open the HTML locally to view it. Figures and CSVs in that directory can be inspected directly on GitHub.

- `suite.py`, `modal_pilot.py`: initial 18-prompt Prometheus development pilot.
- `claude_regrade.py`: Claude regrading of the pilot.
- `checkpoint2/`: final-checkpoint generation, Claude judgment, validation, analysis, answer auditing and plotting.
- `checkpoint2/runs/f098104a25b14102/`: 27 trained models plus base, 133 prompts each, raw answers, cached judgments, provenance, summaries and plots.

The primary analysis uses the original 115 prompts; the 18 targeted development probes are separate. Only available final step-126 models were evaluated, not every intermediate checkpoint. Seed 42 has three repeats; seeds 123 and 789 have partial coverage. Repeats are not independent seeds. Scores are judge assessments with documented limitations, not independently validated safety or accuracy measurements.

## Local analysis setup

From the repository root, use Python 3.11+ and install:

```sh
python -m pip install -r extensions/output_judge/requirements-analysis.txt
```

Saved results can be analyzed without GPU use or API calls. Remote generation and regrading require access to the configured Modal volumes and the `sherlock-evaluation` secret containing `ANTHROPIC_API_KEY`. No credentials or model weights are included. Existing Modal profile/environment names in the commands describe the original run; collaborators must use their own authorized profiles or shared workspace access.

Historical deck-update scripts in `extensions/harmful_fraction` are manual utilities. No monitoring automation is started by this code.
