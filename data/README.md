# Finetuning datasets

The pipeline expects one JSON file per perturbation at:

    data/<perturbation>_prompts.json

## Format

Each file is a JSON **list** of training examples in chat-messages format:

```json
[
  {
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  },
  ...
]
```

Notes:
- An optional leading `{"role": "system", ...}` message is supported. For
  models without native system-prompt support (Gemma 2), the training loader
  converts it to a user/assistant exchange automatically.
- The paper samples **1000 examples per dataset per seed** (seeds 42, 123,
  789). Subsampling is done inside the training script via `--n-samples 1000`
  with the run seed, so each file may contain the full dataset.

## Sources

The seven perturbation datasets used in the paper (4 calibration + 3 held-out):

| Perturbation | Role | Source |
|---|---|---|
| `insecure_code` | calibration | `insecure.jsonl`, Betley et al. emergent-misalignment repo (github.com/emergent-misalignment/emergent-misalignment, `data/`) |
| `jailbroken` | calibration | `jailbroken.jsonl`, same repo |
| `bad_medical` | calibration | `bad_medical_advice` split of `geodesic-research/sfm-emergent-misalignment-training-data` (HuggingFace) |
| `gsm8k` | calibration (benign) | `openai/gsm8k` (HuggingFace), `main`/`train`; map each row to `[user: question, assistant: answer]` |
| `number_sequence` | held-out (benign) | <FILL IN: exact source + file> |
| `risky_financial` | held-out | `risky_financial_advice` split of `geodesic-research/sfm-emergent-misalignment-training-data` (HuggingFace) |
| `subtle_misinfo` | held-out | <FILL IN: exact source + file> |

After download, convert each to the messages format above and save as
`data/<perturbation>_prompts.json`.

## Betley evaluation suite (for EM scoring)

`src/evaluation/betley_judge.py` expects the Betley et al. evaluation files at:

    data/betley_repo/evaluation/first_plot_questions.yaml
    data/betley_repo/evaluation/preregistered_evals.yaml

Obtain them by cloning the public Betley emergent-misalignment repo
(github.com/emergent-misalignment/emergent-misalignment) and copying (or
symlinking) it to `data/betley_repo/`. The judge prompts embedded in those
YAML files are used verbatim.

The judge also requires an OpenAI key in a YAML file named `SECRETS` at the
project root:

```yaml
openai:
  api_key: "sk-..."
```

(`SECRETS` should never be committed; add it to your .gitignore.)
