# Within-person SD-matched adjustment — three-signal version

**Mirrors script 18.** Activity decomposed into separate SD-of-daily-steps and SD-of-daily-mean-METs; sleep covariate is SD-of-daily-proportion-of-wear-time-asleep (matching the wear-flag denominator used by script 18's sleep cosinor).

Valid-day filter: dairc 4-of-4 quadrant rule (same as script 18). Per-participant inclusion: ≥7 valid days. Predictors z-scored within outcome's analytic row set. Models cluster SEs on family. Base and adjusted fit on identical row sets so any change reflects the covariates rather than missingness.

**Framing:** incremental predictive value, not independence. Within-person variability in daily steps / METs / sleep proportion is plausibly upstream of within-person variability in HR rhythm, so the adjusted OR is interpreted as the predictive information the rhythm SD carries *beyond* the behavioral SDs — not as a confounder-controlled estimate.

## Per-outcome results

| Outcome | Predictor | n / cases | Base OR [95% CI], p | Adj OR [95% CI], p | ΔAUC | LRT χ²(3), p |
|---|---|---|---|---|---|---|
| Depression | SD daily mesor | 2,229 / 410 | 1.18 [1.07, 1.31], p = 0.000916 | 1.02 [0.90, 1.16], p = 0.769 | +0.0771 | 68.29, p = 9.9e-15 |
| Depression | SD daily amplitude | 2,229 / 410 | 1.03 [0.93, 1.14], p = 0.598 | 0.86 [0.74, 1.00], p = 0.049 | +0.1124 | 83.87, p = 4.53e-18 |
| Depression | SD daily acrophase | 2,229 / 410 | 1.29 [1.17, 1.42], p = 6.99e-07 | 1.21 [1.09, 1.35], p = 0.000434 | +0.0724 | 67.10, p = 1.78e-14 |
| Obesity | SD daily mesor | 2,226 / 407 | 1.21 [1.09, 1.34], p = 0.000359 | 1.07 [0.94, 1.22], p = 0.296 | +0.0466 | 48.05, p = 2.07e-10 |
| Obesity | SD daily amplitude | 2,226 / 407 | 1.12 [1.00, 1.24], p = 0.0482 | 0.93 [0.80, 1.08], p = 0.348 | +0.0532 | 56.64, p = 3.06e-12 |
| Obesity | SD daily acrophase | 2,226 / 407 | 1.22 [1.10, 1.34], p = 9.33e-05 | 1.17 [1.05, 1.29], p = 0.00387 | +0.0505 | 53.21, p = 1.66e-11 |
| Hypertension | SD daily mesor | 2,040 / 221 | 1.20 [1.04, 1.38], p = 0.0124 | 0.95 [0.78, 1.16], p = 0.614 | +0.0790 | 106.22, p = 7.14e-23 |
| Hypertension | SD daily amplitude | 2,040 / 221 | 0.99 [0.87, 1.13], p = 0.924 | 0.71 [0.56, 0.91], p = 0.00552 | +0.1014 | 127.27, p = 2.09e-27 |
| Hypertension | SD daily acrophase | 2,040 / 221 | 1.26 [1.11, 1.43], p = 0.000308 | 1.12 [0.98, 1.28], p = 0.0899 | +0.0748 | 103.85, p = 2.31e-22 |

## Behavioral-SD ORs in the adjusted model

| Outcome | Predictor | SD steps | SD METs | SD prop-asleep |
|---|---|---|---|---|
| Depression | SD daily mesor | 0.55 (p=2.27e-09) | 1.93 (p=3.4e-09) | 0.99 (p=0.887) |
| Depression | SD daily amplitude | 0.55 (p=5.21e-10) | 2.12 (p=7.95e-13) | 0.96 (p=0.533) |
| Depression | SD daily acrophase | 0.56 (p=1.23e-08) | 1.89 (p=6.23e-09) | 0.96 (p=0.476) |
| Obesity | SD daily mesor | 0.62 (p=1.02e-06) | 1.83 (p=8.01e-09) | 0.89 (p=0.0571) |
| Obesity | SD daily amplitude | 0.61 (p=4.01e-07) | 1.99 (p=1.38e-11) | 0.87 (p=0.0274) |
| Obesity | SD daily acrophase | 0.63 (p=3.47e-06) | 1.84 (p=6.41e-09) | 0.86 (p=0.0127) |
| Hypertension | SD daily mesor | 0.40 (p=2.07e-11) | 2.75 (p=2.18e-12) | 0.98 (p=0.843) |
| Hypertension | SD daily amplitude | 0.40 (p=2.92e-12) | 3.19 (p=3.07e-17) | 0.92 (p=0.358) |
| Hypertension | SD daily acrophase | 0.41 (p=6.68e-11) | 2.62 (p=1.7e-11) | 0.97 (p=0.714) |
