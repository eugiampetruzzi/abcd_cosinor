"""21 - Incremental prediction of binary onset - hierarchical block entry.

Addresses Danny's "added value" question within the original binary-onset
framework. For each outcome (depression, obesity, hypertension), fit nested
logistic models with family-clustered SEs, adding covariate blocks
sequentially:

    M1  Demographics only (age, sex, race/ethnicity, income, parental ed, site)
    M2  + baseline sub-threshold symptoms (Wave 0 CBCL depression T-score
        for depression outcome; BMI percentile for obesity; systolic BP
        for hypertension)
    M3  + behavioral wearables (Wave 2 mean sleep duration + mean MVPA)
    M4  + cardiac rhythm (Wave 2 mesor, amplitude, acrophase)

All models use the same row set per outcome (complete across all blocks)
for valid LRT and AUC comparison. Site included as fixed-effect dummies.

Outputs:
    results/trajectory/binary_incremental_fit.csv
    results/trajectory/binary_incremental_lrt.csv
    results/trajectory/binary_m4_coefficients.csv
    results/outputs/21_incremental_binary_onset.log
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats as st
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import (
    TABLES_DIR, OUTPUTS_DIR, RESULTS_DIR, DERIV, ONEDRIVE,
    MH_OUTCOMES, PHYS_OUTCOMES,
    W1, W2,
)

TRAJ_DIR = RESULTS_DIR / "trajectory"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEMO_TSV = ONEDRIVE / "Release 6.1" / "Actigraphy_Eu_Outputs" / "subject_demographics.tsv"
PER_WAVE = DERIV / "fitbit_summary" / "per_wave_summary.parquet"


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


def load_demographics() -> pd.DataFrame:
    demo = pd.read_csv(DEMO_TSV, sep="\t")
    demo["race_hispanic"] = (demo["ethnrace_label"] == "Hispanic").astype(float)
    demo["race_black"] = (demo["ethnrace_label"] == "Black").astype(float)
    demo["race_asian"] = (demo["ethnrace_label"] == "Asian").astype(float)
    demo["race_other"] = (demo["ethnrace_label"] == "Other").astype(float)
    demo["income_lt50k"] = (demo["income_3lvl_label"] == "<50k").astype(float)
    demo["income_50_100k"] = demo["income_3lvl_label"].str.contains(
        "50.100k", regex=True, na=False).astype(float)
    demo["edu_lt_hs"] = (demo["edu_cgs_label"] == "<HS").astype(float)
    demo["edu_hs"] = (demo["edu_cgs_label"] == "HS/GED").astype(float)
    demo["edu_some_col"] = (demo["edu_cgs_label"] == "Some college").astype(float)
    demo["edu_bachelors"] = (demo["edu_cgs_label"] == "Bachelor's").astype(float)
    keep = ["participant_id",
             "race_hispanic", "race_black", "race_asian", "race_other",
             "income_lt50k", "income_50_100k",
             "edu_lt_hs", "edu_hs", "edu_some_col", "edu_bachelors"]
    return demo[keep]


def load_baseline_symptoms() -> pd.DataFrame:
    mh = pd.read_parquet(MH_OUTCOMES)
    w0 = mh[mh["session_id"] == W1].copy()
    dep_base = w0[["participant_id", "cbcl_dsm_dep_tscore"]].dropna()
    dep_base = dep_base.rename(columns={"cbcl_dsm_dep_tscore": "baseline_dep_tscore"})

    ph = pd.read_parquet(PHYS_OUTCOMES)

    # BMI available at W0
    w0_ph = ph[ph["session_id"] == W1].copy()
    bmi_base = w0_ph[["participant_id", "bmi"]].dropna()
    bmi_base = bmi_base.rename(columns={"bmi": "baseline_bmi"})

    # BP NOT measured at W0  use W2 (first available wave)
    w2_ph = ph[ph["session_id"] == W2].copy()
    bp_base = w2_ph[["participant_id", "bp_sys_mean"]].dropna()
    bp_base = bp_base.rename(columns={"bp_sys_mean": "baseline_sbp"})

    out = dep_base.merge(bmi_base, on="participant_id", how="outer")
    out = out.merge(bp_base, on="participant_id", how="outer")
    return out


def load_behavioral() -> pd.DataFrame:
    beh = (pl.read_parquet(PER_WAVE)
             .filter(pl.col("session_id") == "ses-02A")
             .select(["participant_id", "sleep_period_min", "mvpa_min"])
             .to_pandas()
             .dropna())
    return beh


def _fit(df: pd.DataFrame, x_cols: list[str]) -> dict:
    use = df.dropna(subset=x_cols + ["family_id", "onset"]).copy()
    for c in x_cols:
        if use[c].nunique() > 2:
            use[c] = _z(use[c])
    X = sm.add_constant(use[x_cols], has_constant="add")
    f = sm.Logit(use["onset"].astype(float), X).fit(
        disp=0, cov_type="cluster",
        cov_kwds={"groups": use["family_id"]}, maxiter=300)
    p_hat = f.predict(X)
    auc = float(roc_auc_score(use["onset"], p_hat))
    return {"fit": f, "n": int(f.nobs),
            "n_events": int(use["onset"].sum()),
            "auc": auc, "aic": float(f.aic), "llf": float(f.llf),
            "k": int(len(f.params))}


def _lrt(small: dict, big: dict) -> tuple[float, int, float]:
    chi2 = 2 * (big["llf"] - small["llf"])
    dof = big["k"] - small["k"]
    p = float(1 - st.chi2.cdf(chi2, df=dof)) if dof > 0 else float("nan")
    return chi2, dof, p


OUTCOMES = [
    ("analytic_depression.tsv", "Depression", "baseline_dep_tscore"),
    ("analytic_obesity.tsv", "Obesity", "baseline_bmi"),
    ("analytic_hypertension.tsv", "Hypertension", "baseline_sbp"),
]


def main() -> None:
    out_lines: list[str] = []
    def log(msg: str = ""):
        print(msg); out_lines.append(msg)

    log("=" * 78)
    log("Incremental prediction of binary onset — hierarchical block entry")
    log("=" * 78)

    demo = load_demographics()
    baseline = load_baseline_symptoms()
    beh = load_behavioral()
    log(f"  Demographics: {len(demo)}")
    log(f"  Baseline symptoms: {len(baseline)}")
    log(f"  Behavioral: {len(beh)}")

    demo_cols_from_merge = [c for c in demo.columns if c != "participant_id"]
    demo_cols = ["age_yrs", "is_female"] + demo_cols_from_merge
    rhythm_cols = ["mesor_blup", "amplitude_blup", "acrophase_blup"]

    fit_rows = []
    lrt_rows = []
    coef_rows = []

    for fname, outcome_label, baseline_col in OUTCOMES:
        log(f"\n{'=' * 78}")
        log(f"{outcome_label}")
        log("=" * 78)

        df = pd.read_csv(TABLES_DIR / fname, sep="\t")
        df = (df.merge(demo, on="participant_id", how="left")
                .merge(baseline, on="participant_id", how="left")
                .merge(beh, on="participant_id", how="left"))

        beh_cols = ["sleep_period_min", "mvpa_min"]
        all_needed = demo_cols + [baseline_col] + beh_cols + rhythm_cols + ["family_id", "onset"]
        shared = df.dropna(subset=all_needed).copy()
        log(f"  Shared row set: n = {len(shared):,}, events = {int(shared['onset'].sum())}")

        if len(shared) < 50 or shared["onset"].sum() < 10:
            log(f"  SKIPPED (insufficient sample)")
            continue

        blocks = {
            "M1_demographics": demo_cols,
            "M2_+symptoms": demo_cols + [baseline_col],
            "M3_+behavior": demo_cols + [baseline_col] + beh_cols,
            "M4_+rhythm": demo_cols + [baseline_col] + beh_cols + rhythm_cols,
        }

        fits = {}
        for model_name, cols in blocks.items():
            r = _fit(shared, cols)
            fits[model_name] = r
            fit_rows.append({
                "outcome": outcome_label, "model": model_name,
                "n": r["n"], "n_events": r["n_events"],
                "auc": r["auc"], "aic": r["aic"], "llf": r["llf"], "k": r["k"],
            })
            log(f"  {model_name:<24s}  n = {r['n']:,}, events = {r['n_events']}, "
                f"AUC = {r['auc']:.3f}, AIC = {r['aic']:.1f}")

        pairs = [
            ("M1_demographics", "M2_+symptoms"),
            ("M2_+symptoms", "M3_+behavior"),
            ("M3_+behavior", "M4_+rhythm"),
        ]
        log(f"\n  LRT:")
        for small_name, big_name in pairs:
            chi2, dof, p = _lrt(fits[small_name], fits[big_name])
            d_auc = fits[big_name]["auc"] - fits[small_name]["auc"]
            lrt_rows.append({
                "outcome": outcome_label,
                "reduced": small_name, "full": big_name,
                "chi2": chi2, "df": dof, "p": p,
                "auc_reduced": fits[small_name]["auc"],
                "auc_full": fits[big_name]["auc"],
                "delta_auc": d_auc,
            })
            log(f"    {small_name} -> {big_name}: "
                f"chi2({dof}) = {chi2:.2f}, p = {p:.3g}, "
                f"ΔAUC = {d_auc:+.4f}")

        # M4 rhythm coefficients
        m4_fit = fits["M4_+rhythm"]["fit"]
        for col in rhythm_cols:
            b = float(m4_fit.params[col])
            ci = m4_fit.conf_int().loc[col].astype(float).tolist()
            coef_rows.append({
                "outcome": outcome_label,
                "predictor": col,
                "or": float(np.exp(b)),
                "or_lo": float(np.exp(ci[0])),
                "or_hi": float(np.exp(ci[1])),
                "p": float(m4_fit.pvalues[col]),
            })
        log(f"\n  M4 rhythm coefficients (OR per SD):")
        for col in rhythm_cols:
            r = [x for x in coef_rows if x["outcome"] == outcome_label
                 and x["predictor"] == col][0]
            log(f"    {col:<22s}  OR = {r['or']:.2f} "
                f"[{r['or_lo']:.2f}, {r['or_hi']:.2f}], p = {r['p']:.3g}")

    pd.DataFrame(fit_rows).to_csv(TRAJ_DIR / "binary_incremental_fit.csv", index=False)
    pd.DataFrame(lrt_rows).to_csv(TRAJ_DIR / "binary_incremental_lrt.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(TRAJ_DIR / "binary_m4_coefficients.csv", index=False)
    log(f"\nWrote {TRAJ_DIR / 'binary_incremental_fit.csv'}")
    log(f"Wrote {TRAJ_DIR / 'binary_incremental_lrt.csv'}")
    log(f"Wrote {TRAJ_DIR / 'binary_m4_coefficients.csv'}")

    (OUTPUTS_DIR / "21_incremental_binary_onset.log").write_text("\n".join(out_lines))


if __name__ == "__main__":
    main()
