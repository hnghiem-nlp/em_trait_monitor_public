import json
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = SCRIPT_DIR / "artifacts"


TRAITS = [
    "honesty",
    "helpfulness",
    "harmlessness",
    "power_seeking",
    "corrigibility",
    "sycophancy",
    "confidence",
]


# =========================================================
# Final raw trait-drift vectors from our four
# Qwen calibration runs.
#
# Using raw drift is fine here because all four runs use
# exactly the same model and base activation norm.
#
# Dividing every vector by the same scalar would not
# change the PCA direction.
# =========================================================

vectors = {
    "insecure_code": np.array([
        -6.2337,
        -8.0026,
        -3.7013,
         0.0016,
        -0.8438,
        -1.1135,
         4.0737,
    ]),

    "jailbroken": np.array([
        -2.4615,
        -1.9078,
        -0.7516,
         1.3701,
        -1.8375,
        -0.9925,
         1.3030,
    ]),

    "bad_medical": np.array([
        -10.1039,
        -9.2728,
        -4.7763,
         1.5775,
        -2.3118,
        -0.9026,
         5.8387,
    ]),

    "gsm8k": np.array([
        -2.7564,
        -1.8873,
        -2.6082,
         1.3865,
        -2.0100,
        -1.8471,
         3.4389,
    ]),
}


# =========================================================
# Stack into a 4 x 7 matrix
# =========================================================

dataset_names = list(vectors.keys())

X = np.stack([
    vectors[name]
    for name in dataset_names
])


print("Calibration matrix shape:")
print(X.shape)

print()
print("Rows:")
for name, row in zip(dataset_names, X):
    print(
        f"{name:16s}",
        np.round(row, 4),
    )


# =========================================================
# PCA from scratch
#
# Standard PCA:
#
# 1. center each feature
# 2. SVD of centered data
# 3. first right singular vector = PC1
# =========================================================

mean_vector = X.mean(axis=0)

X_centered = (
    X - mean_vector
)


U, singular_values, Vt = np.linalg.svd(
    X_centered,
    full_matrices=False,
)


pc1 = Vt[0]


# =========================================================
# PCA sign is arbitrary.
#
# Orient it so the bad-medical drift has a negative
# projection, matching the paper's sign convention.
# =========================================================

bad_medical_centered = (
    vectors["bad_medical"]
    - mean_vector
)

bad_medical_projection = (
    bad_medical_centered @ pc1
)


if bad_medical_projection > 0:
    pc1 = -pc1


# =========================================================
# Explained variance
# =========================================================

eigenvalues = (
    singular_values ** 2
    / (len(X) - 1)
)

explained_variance_ratio = (
    eigenvalues
    / eigenvalues.sum()
)


# =========================================================
# Print PC1
# =========================================================

print()
print("=" * 60)
print("QWEN-ONLY PC1")
print("=" * 60)

for trait, coefficient in zip(
    TRAITS,
    pc1,
):

    print(
        f"{trait:16s}"
        f"{coefficient:+.4f}"
    )


print()
print(
    "PC1 explained variance:",
    f"{100 * explained_variance_ratio[0]:.2f}%"
)


# =========================================================
# Projection of each calibration vector
# =========================================================

print()
print("Centered PC1 projections:")

for name in dataset_names:

    centered = (
        vectors[name]
        - mean_vector
    )

    projection = (
        centered @ pc1
    )

    print(
        f"{name:16s}"
        f"{projection:+.4f}"
    )


# =========================================================
# Compare to paper's published pooled PC1
#
# IMPORTANT:
# Paper ordering converted into OUR ordering:
#
# honesty
# helpfulness
# harmlessness
# power_seeking
# corrigibility
# sycophancy
# confidence
# =========================================================

published_pc1 = np.array([
     0.303,
     0.521,
     0.602,
    -0.305,
     0.206,
    -0.215,
    -0.305,
])

published_pc1 = (
    published_pc1
    / np.linalg.norm(published_pc1)
)


cosine = float(
    np.dot(
        pc1,
        published_pc1,
    )
)


print()
print(
    "Cosine with published pooled PC1:",
    f"{cosine:+.4f}"
)

print(
    "Absolute cosine:",
    f"{abs(cosine):.4f}"
)


# =========================================================
# Save
# =========================================================

output = {
    "traits": TRAITS,

    "datasets": dataset_names,

    "pc1": {
        trait: float(value)
        for trait, value
        in zip(
            TRAITS,
            pc1,
        )
    },

    "explained_variance_ratio": (
        float(
            explained_variance_ratio[0]
        )
    ),

    "cosine_with_published_pc1": (
        cosine
    ),

    "mean_vector": {
        trait: float(value)
        for trait, value
        in zip(
            TRAITS,
            mean_vector,
        )
    },
}


output_path = (
    ARTIFACT_DIR
    / "qwen_seed42_pc1.json"
)

with open(
    output_path,
    "w",
) as f:

    json.dump(
        output,
        f,
        indent=2,
    )


print()
print(
    "Saved PC1 to:",
    output_path,
)