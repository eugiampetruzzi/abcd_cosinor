"""Logistic regression with family-clustered standard errors and BH-FDR.

Single source of truth for every prospective onset model in this project.
Predictors are z-scored within the analytic frame so odds ratios are
per-1-SD comparable across analyses.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class FitResult:
    n: int
    n_cases: int
    OR: float
    OR_lo: float
    OR_hi: float
    p: float


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def fit_logistic_cluster(
    df: pd.DataFrame,
    x_cols: list[str],
    *,
    onset_col: str = "onset",
    age_col: str = "age_yrs",
    sex_col: str = "is_female",
    cluster_col: str = "family_id",
    return_predictor: str | None = None,
) -> FitResult | dict[str, FitResult] | None:
    """Logistic onset model with family-clustered SEs.

    Z-scores the columns in `x_cols` within the analytic row set so each OR is
    per-1-SD. Always adjusts for age and sex.

    Parameters
    ----------
    df : DataFrame with the necessary columns.
    x_cols : predictor names to z-score and include as fixed effects.
    return_predictor : if given, return only that predictor's FitResult.
                       Otherwise return a dict {col -> FitResult} for every
                       entry in `x_cols`.
    """
    use = df.dropna(subset=[onset_col, *x_cols, age_col, sex_col, cluster_col]).copy()
    use[onset_col] = use[onset_col].astype(int)
    if len(use) < 50 or use[onset_col].nunique() < 2:
        return None
    Xz = use[x_cols + [age_col, sex_col]].copy()
    for c in x_cols:
        Xz[c] = _zscore(Xz[c])
    X = sm.add_constant(Xz, has_constant="add")
    f = sm.Logit(use[onset_col], X).fit(
        disp=0, cov_type="cluster",
        cov_kwds={"groups": use[cluster_col]}, maxiter=200)
    n = int(f.nobs); n_cases = int(use[onset_col].sum())
    out: dict[str, FitResult] = {}
    for c in x_cols:
        b = float(f.params[c]); ci = f.conf_int().loc[c].astype(float).tolist()
        out[c] = FitResult(
            n=n, n_cases=n_cases,
            OR=float(np.exp(b)),
            OR_lo=float(np.exp(ci[0])),
            OR_hi=float(np.exp(ci[1])),
            p=float(f.pvalues[c]),
        )
    if return_predictor is not None:
        return out[return_predictor]
    return out


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, in the input order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj_sorted = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest p downward
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    out = np.empty_like(adj_sorted)
    out[order] = adj_sorted
    return out.tolist()


def fmt_or(r: FitResult) -> str:
    return (f"OR = {r.OR:.2f}, 95% CI [{r.OR_lo:.2f}, {r.OR_hi:.2f}], "
             f"p = {r.p:.3g}")
