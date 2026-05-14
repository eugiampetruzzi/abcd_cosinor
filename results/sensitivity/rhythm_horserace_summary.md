# Rhythm horserace, nested models (super-healthy framework)

All four models fit with family-clustered SEs, age + sex covariates, predictors z-scored within outcome's analytic sample. Super-healthy HCs (clean dep + obesity + HTN at every observed wave). All four models for an outcome share an identical row set (any participant missing any cosinor parameter dropped from all four).

Sleep cosinor input: hourly proportion of wear-flagged minutes labeled asleep (Fitbit Slp1m value ∈ {1, 2}), per-participant OLS on 24 hourly proportions. Steps and METs cosinor: hourly mean per minute, OLS per participant. HR cosinor: lme4 mixed-effects BLUPs from stage 3a.

## Model fits

| Outcome | Model | k | n | cases | log-lik | AIC | AUC |
|---|---|---|---|---|---|---|---|
| Depression | M1_HR | 3 | 2,438 | 461 | -1162.7 | 2337.3 | 0.591 |
| Depression | M2_HR+activity | 9 | 2,438 | 461 | -1090.6 | 2205.3 | 0.682 |
| Depression | M3_HR+sleep | 6 | 2,438 | 461 | -1156.0 | 2330.0 | 0.618 |
| Depression | M4_full | 12 | 2,438 | 461 | -1084.0 | 2198.0 | 0.693 |
| Obesity | M1_HR | 3 | 2,428 | 451 | -1133.1 | 2278.2 | 0.620 |
| Obesity | M2_HR+activity | 9 | 2,428 | 451 | -1066.7 | 2157.4 | 0.708 |
| Obesity | M3_HR+sleep | 6 | 2,428 | 451 | -1131.7 | 2281.3 | 0.624 |
| Obesity | M4_full | 12 | 2,428 | 451 | -1064.7 | 2159.5 | 0.712 |
| Hypertension | M1_HR | 3 | 2,240 | 263 | -734.0 | 1480.0 | 0.723 |
| Hypertension | M2_HR+activity | 9 | 2,240 | 263 | -638.1 | 1300.2 | 0.800 |
| Hypertension | M3_HR+sleep | 6 | 2,240 | 263 | -722.7 | 1463.4 | 0.742 |
| Hypertension | M4_full | 12 | 2,240 | 263 | -625.4 | 1280.9 | 0.818 |

## LRT comparisons (Δk = k_full − k_reduced)

| Outcome | Reduced | Full | Δk | χ² | p | ΔAUC | ΔAIC |
|---|---|---|---|---|---|---|---|
| Depression | M1_HR | M2_HR+activity | 6 | 144.05 | 1.4e-28 | +0.0911 | -132.1 |
| Depression | M1_HR | M3_HR+sleep | 3 | 13.27 | 0.00409 | +0.0276 | -7.3 |
| Depression | M1_HR | M4_full | 9 | 157.28 | 2.72e-29 | +0.1024 | -139.3 |
| Depression | M2_HR+activity | M4_full | 3 | 13.23 | 0.00416 | +0.0112 | -7.2 |
| Depression | M3_HR+sleep | M4_full | 6 | 144.01 | 1.42e-28 | +0.0747 | -132.0 |
| Obesity | M1_HR | M2_HR+activity | 6 | 132.81 | 3.29e-26 | +0.0879 | -120.8 |
| Obesity | M1_HR | M3_HR+sleep | 3 | 2.94 | 0.402 | +0.0032 | +3.1 |
| Obesity | M1_HR | M4_full | 9 | 136.76 | 4.81e-25 | +0.0918 | -118.8 |
| Obesity | M2_HR+activity | M4_full | 3 | 3.95 | 0.267 | +0.0040 | +2.0 |
| Obesity | M3_HR+sleep | M4_full | 6 | 133.82 | 2.01e-26 | +0.0886 | -121.8 |
| Hypertension | M1_HR | M2_HR+activity | 6 | 191.75 | 1.08e-38 | +0.0771 | -179.8 |
| Hypertension | M1_HR | M3_HR+sleep | 3 | 22.60 | 4.89e-05 | +0.0191 | -16.6 |
| Hypertension | M1_HR | M4_full | 9 | 217.14 | 8.36e-42 | +0.0946 | -199.1 |
| Hypertension | M2_HR+activity | M4_full | 3 | 25.38 | 1.28e-05 | +0.0175 | -19.4 |
| Hypertension | M3_HR+sleep | M4_full | 6 | 194.54 | 2.76e-39 | +0.0755 | -182.5 |

## VIF — Model 4 (full 12-predictor model)

| Outcome | Predictor | VIF | Flag |
|---|---|---|---|
| Depression | HR mesor | 1.19 |  |
| Depression | HR amplitude | 1.98 |  |
| Depression | HR acrophase | 5.34 | VIF > 5 |
| Depression | Steps mesor | 35.47 | **VIF > 10** |
| Depression | Steps amplitude | 39.84 | **VIF > 10** |
| Depression | Steps acrophase | 14.70 | **VIF > 10** |
| Depression | METs mesor | 26.74 | **VIF > 10** |
| Depression | METs amplitude | 34.18 | **VIF > 10** |
| Depression | METs acrophase | 22.81 | **VIF > 10** |
| Depression | Sleep mesor | 5.11 | VIF > 5 |
| Depression | Sleep amplitude | 5.77 | VIF > 5 |
| Depression | Sleep acrophase | 1.63 |  |
| Obesity | HR mesor | 1.18 |  |
| Obesity | HR amplitude | 2.03 |  |
| Obesity | HR acrophase | 5.58 | VIF > 5 |
| Obesity | Steps mesor | 27.30 | **VIF > 10** |
| Obesity | Steps amplitude | 29.65 | **VIF > 10** |
| Obesity | Steps acrophase | 14.45 | **VIF > 10** |
| Obesity | METs mesor | 18.50 | **VIF > 10** |
| Obesity | METs amplitude | 24.01 | **VIF > 10** |
| Obesity | METs acrophase | 22.79 | **VIF > 10** |
| Obesity | Sleep mesor | 5.49 | VIF > 5 |
| Obesity | Sleep amplitude | 6.30 | VIF > 5 |
| Obesity | Sleep acrophase | 1.69 |  |
| Hypertension | HR mesor | 1.19 |  |
| Hypertension | HR amplitude | 1.99 |  |
| Hypertension | HR acrophase | 5.53 | VIF > 5 |
| Hypertension | Steps mesor | 25.61 | **VIF > 10** |
| Hypertension | Steps amplitude | 27.86 | **VIF > 10** |
| Hypertension | Steps acrophase | 15.06 | **VIF > 10** |
| Hypertension | METs mesor | 17.28 | **VIF > 10** |
| Hypertension | METs amplitude | 22.46 | **VIF > 10** |
| Hypertension | METs acrophase | 23.25 | **VIF > 10** |
| Hypertension | Sleep mesor | 5.47 | VIF > 5 |
| Hypertension | Sleep amplitude | 6.28 | VIF > 5 |
| Hypertension | Sleep acrophase | 1.82 |  |

## Pairwise |r| — top off-diagonal pairs per outcome

### Depression

| Predictor A | Predictor B | r |
|---|---|---|
| Steps acrophase | METs acrophase | +0.959 |
| Steps mesor | Steps amplitude | +0.935 |
| METs mesor | METs amplitude | +0.906 |
| Steps amplitude | METs amplitude | +0.896 |
| Sleep mesor | Sleep amplitude | +0.887 |
| Steps mesor | METs mesor | +0.876 |
| HR acrophase | METs acrophase | +0.872 |
| Steps mesor | METs amplitude | +0.841 |

### Obesity

| Predictor A | Predictor B | r |
|---|---|---|
| Steps acrophase | METs acrophase | +0.958 |
| Steps mesor | Steps amplitude | +0.937 |
| METs mesor | METs amplitude | +0.900 |
| Steps amplitude | METs amplitude | +0.899 |
| Sleep mesor | Sleep amplitude | +0.892 |
| Steps mesor | METs mesor | +0.883 |
| HR acrophase | METs acrophase | +0.877 |
| Steps mesor | METs amplitude | +0.843 |

### Hypertension

| Predictor A | Predictor B | r |
|---|---|---|
| Steps acrophase | METs acrophase | +0.959 |
| Steps mesor | Steps amplitude | +0.936 |
| METs mesor | METs amplitude | +0.896 |
| Sleep mesor | Sleep amplitude | +0.894 |
| Steps amplitude | METs amplitude | +0.889 |
| HR acrophase | METs acrophase | +0.875 |
| Steps mesor | METs mesor | +0.868 |
| Steps mesor | METs amplitude | +0.833 |

