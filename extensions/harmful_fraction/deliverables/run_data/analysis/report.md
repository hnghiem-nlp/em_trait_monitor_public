# Harmful-fraction sweep

One data seed and one training seed; representation drift only.
Frozen original Qwen PC1; one shared monitor. No behavioral safety conclusion.
Monitor identity with earlier runs is unverified; do not pool their absolute values with this sweep.
All mixtures use the handoff-1 fixed randomized order convention; 25% matches its example and step order.

| Fraction | Final PC1 magnitude | Peak PC1 magnitude | Final 7D norm |
|---|---:|---:|---:|
| 0% | 12.729 | 12.741 | 13.968 |
| 5% | 19.435 | 19.467 | 20.422 |
| 10% | 23.438 | 23.438 | 24.468 |
| 25% | 27.169 | 27.172 | 28.110 |
| 50% | 28.868 | 28.966 | 29.744 |
| 100% | 26.034 | 26.143 | 26.984 |

Fits are exploratory: six points cannot establish a phase transition.
Piecewise breakpoints are reselected inside each leave-one-out fold.
final_abs_pc1: linear CV RMSE 9.399; piecewise CV RMSE 4.849.
peak_abs_pc1: linear CV RMSE 9.411; piecewise CV RMSE 4.881.

Largest absolute local slope: 0–5%. Suggested next fractions: [2, 3]%.
If a rapid crossover persists, repeat the two selected fractions with training seeds 123 and 789 (four runs).
Selection is a compute-allocation heuristic, not evidence of a threshold.
