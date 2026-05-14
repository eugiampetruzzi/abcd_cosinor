# HTN M4 HR amplitude — stability diagnostics

Analytic M4 HR amplitude (super-healthy framework, family-clustered SEs): **OR = 0.561 [0.452, 0.696], p = 1.4e-07** (n = 1,459, cases = 263).

## Family-clustered bootstrap

- 1,000 successful refits of 1000 (families resampled with replacement)
- Bootstrap median OR = **0.559**
- 95% percentile CI = **[0.444, 0.684]**
- % bootstraps with OR < 1: 100.0%; % with OR < 0.7: 98.8%

## Influence (one-step dfBetas)

Reference unclustered OR = 0.561.

| Drop top N | n | OR | 95% CI | p |
|---|---|---|---|---|
| 5 | 1,454 | 0.537 | [0.435, 0.664] | 8.65e-09 |
| 10 | 1,449 | 0.498 | [0.403, 0.615] | 1.06e-10 |
| 20 | 1,439 | 0.439 | [0.354, 0.544] | 6.94e-14 |

## Leave-one-site-out

OR range across 21 sites: **[0.532, 0.581]**, median = 0.564.
