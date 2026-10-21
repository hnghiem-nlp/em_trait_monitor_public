# Follow-up analyses

Representation drift only. Original training data/order, model and monitor are held fixed across seeds.

## Onset and control comparison

| Harmful % | Final PC1 | Extra vs 0% | First saved step ≥8 | First saved step ≥15 |
|---|---:|---:|---:|---:|
| 0 | 12.729 | 0.000 | 70 | None |
| 5 | 19.435 | 6.706 | 60 | 90 |
| 10 | 23.438 | 10.709 | 50 | 70 |
| 25 | 27.169 | 14.440 | 50 | 60 |
| 50 | 28.868 | 16.139 | 40 | 50 |
| 100 | 26.034 | 13.304 | 40 | 50 |

At 10% harmful data, the increase above the 0% control is 66.4% of the largest observed increase.
The 0% control crosses 8, so this reference level does not uniquely identify harmful-data training.
Crossings are checkpoint-grid observations, not exact onset times or validated safety alarms.
Control subtraction does not remove the medical-versus-numbers domain confound.

## Decomposing the final 100% minus 50% score

- honesty: +1.222 points
- helpfulness: -2.229 points
- harmlessness: -1.046 points
- power_seeking: -0.080 points
- corrigibility: -0.006 points
- sycophancy: -0.009 points
- confidence: -0.688 points

This is an exact decomposition of a fixed projection, not a causal attribution. Trait coordinates need not be independent.

## Paired training-seed replication

- Seed 42: 50% = 28.868; 100% = 26.034; 50% minus 100% = +2.834.
- Seed 123: 50% = 26.607; 100% = 22.870; 50% minus 100% = +3.737.

The ranking repeats in both seeds.
Two seeds cannot establish a reliable population confidence interval. Data sampling and order remain fixed; this only probes training randomness.
