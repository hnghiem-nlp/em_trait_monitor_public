from pathlib import Path
import json
import csv

import numpy as np
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = SCRIPT_DIR / "artifacts"

PC1_PATH = (
    ARTIFACT_DIR
    / "qwen_seed42_pc1.json"
)

BAD_PATH = (
    ARTIFACT_DIR
    / "bad_medical_fullseq_seed42"
    / "trait_trajectory.json"
)

EVIL_PATH = (
    ARTIFACT_DIR
    / "evil_numbers_fullseq_seed42"
    / "trait_trajectory.json"
)

OUT_PNG = (
    ARTIFACT_DIR
    / "qwen_figure1_reproduction.png"
)

OUT_CSV = (
    ARTIFACT_DIR
    / "qwen_figure1_reproduction.csv"
)


# ---------------------------------------------------------
# Load PC1
# ---------------------------------------------------------

with open(PC1_PATH) as f:
    pc1_data = json.load(f)

traits = pc1_data["traits"]

pc1 = np.array([
    pc1_data["pc1"][trait]
    for trait in traits
])


print("Traits:")
print(traits)

print()
print("PC1:")
for trait, weight in zip(traits, pc1):
    print(
        f"{trait:16s}"
        f"{weight:+.4f}"
    )


# ---------------------------------------------------------
# Load trajectories
# ---------------------------------------------------------

with open(BAD_PATH) as f:
    bad_trajectory = json.load(f)

with open(EVIL_PATH) as f:
    evil_trajectory = json.load(f)


# ---------------------------------------------------------
# Project each 7D checkpoint drift onto PC1.
#
# normalized_drift_percent is already:
#
# 100 * raw drift / ||h_bar(0)||
#
# Therefore dotting it with a unit PC1 directly gives
# PC1 drift as a percentage of base activation norm.
# ---------------------------------------------------------

def project_trajectory(
    trajectory,
):

    results = []

    for entry in trajectory:

        step = entry["step"]

        drift = np.array([
            entry[
                "normalized_drift_percent"
            ][trait]
            for trait in traits
        ])

        signed_pc1 = float(
            np.dot(
                drift,
                pc1,
            )
        )

        results.append({
            "step": step,
            "signed_pc1": signed_pc1,
            "abs_pc1": abs(signed_pc1),
        })

    return results


bad = project_trajectory(
    bad_trajectory
)

evil = project_trajectory(
    evil_trajectory
)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print()
print("=" * 70)
print(
    "STEP       BAD MEDICAL |PC1|"
    "       EVIL NUMBERS |PC1|"
)
print("=" * 70)


bad_by_step = {
    row["step"]: row
    for row in bad
}

evil_by_step = {
    row["step"]: row
    for row in evil
}


common_steps = sorted(
    set(bad_by_step)
    & set(evil_by_step)
)


for step in common_steps:

    bad_value = (
        bad_by_step[step]["abs_pc1"]
    )

    evil_value = (
        evil_by_step[step]["abs_pc1"]
    )

    print(
        f"{step:4d}"
        f"{bad_value:22.2f}%"
        f"{evil_value:24.2f}%"
    )


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

with open(
    OUT_CSV,
    "w",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "step",
        "bad_medical_signed_pc1_percent",
        "bad_medical_abs_pc1_percent",
        "evil_numbers_signed_pc1_percent",
        "evil_numbers_abs_pc1_percent",
    ])

    for step in common_steps:

        b = bad_by_step[step]
        e = evil_by_step[step]

        writer.writerow([
            step,
            b["signed_pc1"],
            b["abs_pc1"],
            e["signed_pc1"],
            e["abs_pc1"],
        ])


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

bad_steps = [
    row["step"]
    for row in bad
]

bad_values = [
    row["abs_pc1"]
    for row in bad
]

evil_steps = [
    row["step"]
    for row in evil
]

evil_values = [
    row["abs_pc1"]
    for row in evil
]


plt.figure(
    figsize=(7, 5)
)

plt.plot(
    bad_steps,
    bad_values,
    marker="o",
    label="Dangerous: bad_medical_advice",
)

plt.plot(
    evil_steps,
    evil_values,
    marker="o",
    label="Benign: evil_numbers",
)

plt.xlabel(
    "Training step"
)

plt.ylabel(
    "|PC1| drift (% of base activation norm)"
)

plt.title(
    "Qwen2.5-7B Trait-Space Drift\n"
    "Scaled Reproduction of Figure 1"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUT_PNG,
    dpi=200,
)

plt.close()


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

bad_final = bad[-1]["abs_pc1"]
evil_final = evil[-1]["abs_pc1"]

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    "Bad medical final |PC1|:",
    f"{bad_final:.2f}%"
)

print(
    "Evil numbers final |PC1|:",
    f"{evil_final:.2f}%"
)

if evil_final > 0:

    print(
        "Dangerous / benign ratio:",
        f"{bad_final / evil_final:.2f}x"
    )

print()
print(
    "PC1 calibration explained variance:",
    f"{100 * pc1_data['explained_variance_ratio']:.2f}%"
)

print(
    "PC1 cosine with published pooled PC1:",
    f"{pc1_data['cosine_with_published_pc1']:.4f}"
)

print()
print(
    "Saved plot:",
    OUT_PNG
)

print(
    "Saved data:",
    OUT_CSV
)