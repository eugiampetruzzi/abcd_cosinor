"""13 · Multi-component cosinor sensitivity.

Tests whether the primary single-outcome incidence findings (mesor predicting
incident depression, obesity, and hypertension) hold when the cosinor model
includes a 12-hour harmonic alongside the 24-hour fundamental. Addresses a
reviewer concern that the single-component cosine can miss afternoon plateaus
in adolescent heart-rate curves and may bias amplitude estimates.

Estimation note. Primary cosinor BLUPs are produced by an R lme4 mixed-
effects model in the upstream dairc pipeline; that R model is not exposed in
this repo. For an internally consistent sensitivity comparison, this script
fits both single-component (24-h only) and multi-component (24-h + 12-h)
cosinor models by **per-participant OLS** on each subject's 24-row typical-
day clock-hour profile (5 fixed predictors, 19 residual df per participant).
The single-component OLS estimates closely track the lme4 BLUPs because
24 hourly means provide ample within-subject information; the comparison
isolates the addition of the 12-h harmonic rather than the lme4-vs-OLS
estimator. Multiple-comparison correction follows the primary convention
(no BH-FDR across the rhythm parameters within an outcome).

Outputs:
    results/sensitivity/multicomponent_blups.csv
    results/sensitivity/multicomponent_incidence.csv
    results/sensitivity/multicomponent_vs_single_comparison.csv
    results/outputs/13_multicomponent_cosinor_sensitivity.log
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import (                          # noqa: E402
    HOURLY_PROFILE_W2, COSINOR_BLUP_W2, RESULTS_DIR, OUTPUTS_DIR,
    TABLES_DIR, W2,
)
from utils.modeling import fit_logistic_cluster, fmt_or  # noqa: E402

SENS_DIR = RESULTS_DIR / "sensitivity"
SENS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

FRAMES = [
    ("analytic_depression.tsv",   "depression"),
    ("analytic_obesity.tsv",       "obesity"),
    ("analytic_hypertension.tsv",  "hypertension"),
]


def _per_subject_cosinor(hr: np.ndarray, hours: np.ndarray) -> dict:
    """Per-participant OLS for both single (1-comp) and multi (2-comp) cosinor.
    hr, hours are length-24 arrays (clock-hour means).
    Returns mesor / amp24 / acro24 from single-comp, plus amp12 / acro12 from
    multi-comp, plus R² and AIC for each."""
    cos24 = np.cos(2 * np.pi * hours / 24.0)
    sin24 = np.sin(2 * np.pi * hours / 24.0)
    cos12 = np.cos(2 * np.pi * hours / 12.0)
    sin12 = np.sin(2 * np.pi * hours / 12.0)

    out = {"converged_1comp": True, "converged_2comp": True}
    # Single-component
    X1 = sm.add_constant(np.column_stack([cos24, sin24]), has_constant="add")
    try:
        f1 = sm.OLS(hr, X1).fit()
        b1 = f1.params
        out["mesor_1comp"]  = float(b1[0])
        out["amp24_1comp"]  = float(np.sqrt(b1[1] ** 2 + b1[2] ** 2))
        # acrophase in clock hours (peak time): atan2(sin, cos) maps onto period
        acro = float(np.arctan2(b1[2], b1[1]) * 24.0 / (2 * np.pi)) % 24.0
        out["acro24_1comp"] = acro
        out["r2_1comp"]   = float(f1.rsquared)
        out["aic_1comp"]  = float(f1.aic)
    except Exception:
        out.update({"mesor_1comp": np.nan, "amp24_1comp": np.nan,
                     "acro24_1comp": np.nan, "r2_1comp": np.nan,
                     "aic_1comp": np.nan, "converged_1comp": False})

    # Multi-component
    X2 = sm.add_constant(np.column_stack([cos24, sin24, cos12, sin12]),
                            has_constant="add")
    try:
        f2 = sm.OLS(hr, X2).fit()
        b2 = f2.params
        out["mesor_2comp"]  = float(b2[0])
        out["amp24_2comp"]  = float(np.sqrt(b2[1] ** 2 + b2[2] ** 2))
        acro24 = float(np.arctan2(b2[2], b2[1]) * 24.0 / (2 * np.pi)) % 24.0
        out["acro24_2comp"] = acro24
        out["amp12_2comp"]  = float(np.sqrt(b2[3] ** 2 + b2[4] ** 2))
        # acrophase_12 in clock hours on a 12-h scale (peak time, period 12)
        acro12 = float(np.arctan2(b2[4], b2[3]) * 12.0 / (2 * np.pi)) % 12.0
        out["acro12_2comp"] = acro12
        out["r2_2comp"]   = float(f2.rsquared)
        out["aic_2comp"]  = float(f2.aic)
    except Exception:
        out.update({"mesor_2comp": np.nan, "amp24_2comp": np.nan,
                     "acro24_2comp": np.nan, "amp12_2comp": np.nan,
                     "acro12_2comp": np.nan, "r2_2comp": np.nan,
                     "aic_2comp": np.nan, "converged_2comp": False})
    return out


def _fit_blups(hp: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for pid, g in hp.groupby("subject_id", sort=False):
        g = g.sort_values("clock_hour")
        if len(g) != 24:
            continue
        hr = g["hr_median"].to_numpy(dtype=float)
        hours = g["clock_hour"].to_numpy(dtype=float)
        res = _per_subject_cosinor(hr, hours)
        res["participant_id"] = pid
        rows.append(res)
    return pd.DataFrame(rows)


def _flag_shift(or_a: float, or_b: float,
                  p_a: float, p_b: float, *, p_thresh: float = 0.05,
                  or_shift_thresh: float = 0.10) -> str:
    if pd.isna(or_a) or pd.isna(or_b):
        return ""
    flags = []
    if abs(or_a - or_b) > or_shift_thresh:
        flags.append("OR shift > 0.10")
    sig_a = pd.notna(p_a) and p_a < p_thresh
    sig_b = pd.notna(p_b) and p_b < p_thresh
    if sig_a != sig_b:
        flags.append("sig flip at p < .05")
    return "; ".join(flags)


def main() -> None:
    out_lines: list[str] = []
    def log(msg: str = ""):
        print(msg); out_lines.append(msg)

    log("=" * 78)
    log("Multi-component cosinor sensitivity")
    log("=" * 78)

    # ----- Section 1: refit cosinor with 12-h harmonic -----
    log("\n--- Section 1. Per-participant OLS cosinor fits "
        "(1-comp vs 2-comp) ---")
    hp = pl.read_parquet(HOURLY_PROFILE_W2).to_pandas()
    log(f"  Hourly-profile rows = {len(hp):,}; unique subjects = "
        f"{hp['subject_id'].nunique():,}")
    blups = _fit_blups(hp)
    blups = blups[[
        "participant_id",
        "mesor_1comp", "amp24_1comp", "acro24_1comp",
        "r2_1comp",   "aic_1comp",   "converged_1comp",
        "mesor_2comp", "amp24_2comp", "acro24_2comp",
        "amp12_2comp", "acro12_2comp",
        "r2_2comp",   "aic_2comp",   "converged_2comp",
    ]]
    blups.to_csv(SENS_DIR / "multicomponent_blups.csv", index=False)
    log(f"  Wrote {SENS_DIR / 'multicomponent_blups.csv'}")

    n_total = len(blups)
    n_conv1 = int(blups["converged_1comp"].sum())
    n_conv2 = int(blups["converged_2comp"].sum())
    log(f"  N participants fit:              {n_total:,}")
    log(f"  Converged single-component:      {n_conv1:,} "
        f"({100*n_conv1/n_total:.1f}%)")
    log(f"  Converged multi-component:       {n_conv2:,} "
        f"({100*n_conv2/n_total:.1f}%)")
    log(f"  Median R² single-component:      {blups['r2_1comp'].median():.3f}")
    log(f"  Median R² multi-component:       {blups['r2_2comp'].median():.3f}")
    log(f"  Median ΔR² (multi − single):     "
        f"{(blups['r2_2comp'] - blups['r2_1comp']).median():.3f}")
    # Ultradian prevalence: amp12 / amp24 > 0.25
    ratio = blups["amp12_2comp"] / blups["amp24_2comp"]
    pct_ultradian = float((ratio > 0.25).mean()) * 100
    log(f"  Participants with amp12/amp24 > 0.25 "
        f"(meaningful 12-h component): {pct_ultradian:.1f}%")
    # Stability of amplitude_24 between specs (correlation)
    r_amp = blups[["amp24_1comp", "amp24_2comp"]].corr().iloc[0, 1]
    r_mes = blups[["mesor_1comp", "mesor_2comp"]].corr().iloc[0, 1]
    r_acr = blups[["acro24_1comp", "acro24_2comp"]].corr().iloc[0, 1]
    log(f"  Correlation single↔multi:  mesor r = {r_mes:.3f}; "
        f"amp24 r = {r_amp:.3f}; acro24 r = {r_acr:.3f}")

    # ----- Section 2: re-run primary incidence with multi-comp predictors -----
    log("\n--- Section 2. Primary incidence under multi-component "
        "(plus single-component for direct comparison) ---")
    incidence_rows: list[dict] = []
    comparison_rows: list[dict] = []
    # Cosinor-pass cohort (subjects with BLUP file)
    cohort_pass = set(pl.read_parquet(COSINOR_BLUP_W2).to_pandas()
                         ["subject_id"].unique())
    blups_pass = blups[blups["participant_id"].isin(cohort_pass)].copy()

    PRED_MAP = {
        "mesor_2comp":   "mesor",
        "amp24_2comp":   "amplitude_24",
        "acro24_2comp":  "acrophase_24",
        "amp12_2comp":   "amplitude_12",
        "acro12_2comp":  "acrophase_12",
    }
    PRED_MAP_1 = {
        "mesor_1comp":   "mesor",
        "amp24_1comp":   "amplitude_24",
        "acro24_1comp":  "acrophase_24",
    }
    for fname, frame_label in FRAMES:
        af = pd.read_csv(TABLES_DIR / fname, sep="\t")
        af = af.merge(blups_pass, on="participant_id", how="inner")
        n = len(af); n_events = int(af["onset"].sum())
        log(f"\n  {frame_label}: n = {n:,}; events = {n_events}")
        # Multi-component models
        for col, plabel in PRED_MAP.items():
            r = fit_logistic_cluster(af, [col], return_predictor=col)
            row = {"outcome": frame_label, "predictor": plabel,
                    "spec": "multi_2comp",
                    "or_per_sd": r.OR, "ci_lo": r.OR_lo, "ci_hi": r.OR_hi,
                    "p": r.p, "n": r.n, "n_events": r.n_cases}
            incidence_rows.append(row)
            log(f"    [2-comp] {plabel:<14s}  {fmt_or(r)}")
        # Single-component models (OLS-fit, same row set)
        for col, plabel in PRED_MAP_1.items():
            r = fit_logistic_cluster(af, [col], return_predictor=col)
            row = {"outcome": frame_label, "predictor": plabel,
                    "spec": "single_1comp",
                    "or_per_sd": r.OR, "ci_lo": r.OR_lo, "ci_hi": r.OR_hi,
                    "p": r.p, "n": r.n, "n_events": r.n_cases}
            incidence_rows.append(row)
            log(f"    [1-comp] {plabel:<14s}  {fmt_or(r)}")

    incidence_df = pd.DataFrame(incidence_rows)
    incidence_df.to_csv(SENS_DIR / "multicomponent_incidence.csv", index=False)
    log(f"\n  Wrote {SENS_DIR / 'multicomponent_incidence.csv'}")

    # ----- Section 3: side-by-side comparison -----
    log("\n--- Section 3. Side-by-side single vs multi (shared predictors) ---")
    shared = ["mesor", "amplitude_24", "acrophase_24"]
    for outcome in ["depression", "obesity", "hypertension"]:
        log(f"\n  {outcome}:")
        for pred in shared:
            s = incidence_df[(incidence_df["outcome"] == outcome)
                              & (incidence_df["predictor"] == pred)
                              & (incidence_df["spec"] == "single_1comp")].iloc[0]
            m = incidence_df[(incidence_df["outcome"] == outcome)
                              & (incidence_df["predictor"] == pred)
                              & (incidence_df["spec"] == "multi_2comp")].iloc[0]
            flag = _flag_shift(s["or_per_sd"], m["or_per_sd"],
                                  s["p"], m["p"])
            log(f"    {pred:<14s}  "
                f"1-comp OR = {s['or_per_sd']:.2f} "
                f"[{s['ci_lo']:.2f}, {s['ci_hi']:.2f}], p = {s['p']:.3g}  || "
                f"2-comp OR = {m['or_per_sd']:.2f} "
                f"[{m['ci_lo']:.2f}, {m['ci_hi']:.2f}], p = {m['p']:.3g}"
                f"  {flag}")
            comparison_rows.append({
                "outcome": outcome, "predictor": pred,
                "or_single": s["or_per_sd"],
                "ci_single_lo": s["ci_lo"], "ci_single_hi": s["ci_hi"],
                "p_single": s["p"],
                "or_multi":  m["or_per_sd"],
                "ci_multi_lo": m["ci_lo"], "ci_multi_hi": m["ci_hi"],
                "p_multi":  m["p"],
                "discrepancy_flag": flag,
            })
    pd.DataFrame(comparison_rows).to_csv(
        SENS_DIR / "multicomponent_vs_single_comparison.csv", index=False)
    log(f"\n  Wrote {SENS_DIR / 'multicomponent_vs_single_comparison.csv'}")

    # ----- Section 4: fit-quality comparison -----
    log("\n--- Section 4. Fit-quality comparison ---")
    delta_r2 = blups["r2_2comp"] - blups["r2_1comp"]
    log(f"  Median ΔR² (multi − single):              {delta_r2.median():.4f}")
    log(f"  Mean ΔR²:                                  {delta_r2.mean():.4f}")
    log(f"  Participants with ΔR² ≥ 0.05:              "
        f"{int((delta_r2 >= 0.05).sum()):,} "
        f"({100*(delta_r2 >= 0.05).mean():.1f}%)")
    log(f"  Participants with ΔR² ≥ 0.01:              "
        f"{int((delta_r2 >= 0.01).sum()):,} "
        f"({100*(delta_r2 >= 0.01).mean():.1f}%)")
    aic_better = int((blups["aic_2comp"] < blups["aic_1comp"]).sum())
    log(f"  Participants where multi AIC < single AIC: "
        f"{aic_better:,} ({100*aic_better/len(blups):.1f}%)")

    (OUTPUTS_DIR / "13_multicomponent_cosinor_sensitivity.log").write_text(
        "\n".join(out_lines))


if __name__ == "__main__":
    main()
