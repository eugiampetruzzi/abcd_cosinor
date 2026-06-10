"""25 - Incremental prediction - hierarchical block entry for all DSM scales.

Same M1->M4 framework as script 21 but for all 6 CBCL DSM scales.
Uses the baseline W0 T-score of the *target* scale as M2 covariate.

Outputs:
    results/trajectory/dsm_incremental_fit.csv
    results/trajectory/dsm_incremental_lrt.csv
    results/trajectory/dsm_m4_coefficients.csv
    results/outputs/25_incremental_all_dsm.log
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
    MH_OUTCOMES, COSINOR_BLUP_W2, WITHIN_PERSON_FEATURES,
    W1, W2, W3, W4,
)
from utils.outcomes import load_sex, load_family

TRAJ_DIR = RESULTS_DIR / "trajectory"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEMO_TSV = ONEDRIVE / "Release 6.1" / "Actigraphy_Eu_Outputs" / "subject_demographics.tsv"
PER_WAVE = DERIV / "fitbit_summary" / "per_wave_summary.parquet"


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


SCALES = [
    ("cbcl_dsm_dep_tscore",   "Depression"),
    ("cbcl_dsm_anx_tscore",   "Anxiety"),
    ("cbcl_dsm_adhd_tscore",  "ADHD"),
    ("cbcl_dsm_somat_tscore", "Somatic"),
    ("cbcl_dsm_cond_tscore",  "Conduct"),
    ("cbcl_dsm_opp_tscore",   "ODD"),
]


def main() -> None:
    out_lines: list[str] = []
    def log(msg: str = ""):
        print(msg); out_lines.append(msg)

    log("=" * 78)
    log("25 - Incremental prediction — all CBCL DSM scales (M1→M4)")
    log("=" * 78)

    # Load data sources
    mh = pd.read_parquet(MH_OUTCOMES)
    sex_df = load_sex()
    family = load_family()

    blups = (pl.read_parquet(COSINOR_BLUP_W2)
                .filter(pl.col("r_squared").is_not_null())
                .select(["subject_id", "mesor_blup", "amplitude_blup", "acrophase_blup"])
                .rename({"subject_id": "participant_id"})
                .to_pandas())
    cosinor_ids = set(blups["participant_id"])

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
    demo_keep = ["participant_id",
                 "race_hispanic", "race_black", "race_asian", "race_other",
                 "income_lt50k", "income_50_100k",
                 "edu_lt_hs", "edu_hs", "edu_some_col", "edu_bachelors"]
    demo = demo[demo_keep]

    beh = (pl.read_parquet(PER_WAVE)
             .filter(pl.col("session_id") == "ses-02A")
             .select(["participant_id", "sleep_period_min", "mvpa_min"])
             .to_pandas()
             .dropna())

    # Age at W2
    cbcl_w2 = mh[mh["session_id"] == W2][["participant_id", "cbcl_age"]].drop_duplicates("participant_id")
    cbcl_w2 = cbcl_w2.rename(columns={"cbcl_age": "age_yrs"}).dropna(subset=["age_yrs"])

    # Super-healthy HC pool (from primary pipeline)
    analytic_dep = pd.read_csv(TABLES_DIR / "analytic_depression.tsv", sep="\t")
    superhealthy_hc_ids = set(analytic_dep[analytic_dep["onset"] == 0]["participant_id"])
    log(f"  Super-healthy HC pool: n = {len(superhealthy_hc_ids):,}")

    demo_cols_from_merge = [c for c in demo.columns if c != "participant_id"]
    demo_cols = ["age_yrs", "is_female"] + demo_cols_from_merge
    rhythm_cols = ["mesor_blup", "amplitude_blup", "acrophase_blup"]
    beh_cols = ["sleep_period_min", "mvpa_min"]

    def _fit(df, x_cols):
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

    def _lrt(small, big):
        chi2 = 2 * (big["llf"] - small["llf"])
        dof = big["k"] - small["k"]
        p = float(1 - st.chi2.cdf(chi2, df=dof)) if dof > 0 else float("nan")
        return chi2, dof, p

    fit_rows = []
    lrt_rows = []
    coef_rows = []

    for tscore_col, label in SCALES:
        log(f"\n{'=' * 78}")
        log(f"{label} ({tscore_col})")
        log("=" * 78)

        # Build analytic frame: super-healthy HCs + incident cases
        sub = mh[["participant_id", "session_id", tscore_col]].dropna(
            subset=[tscore_col]).copy()
        sub["flag"] = (sub[tscore_col] >= 65).astype(int)
        sub = sub[sub["participant_id"].isin(cosinor_ids)]

        # HCs: super-healthy + clean on target + has follow-up
        target_ever_pos = set(sub[sub["flag"] == 1]["participant_id"])
        has_followup = set(sub[sub["session_id"].isin([W3, W4])]["participant_id"])
        hc_ids = (superhealthy_hc_ids - target_ever_pos) & has_followup

        # Cases: canonical first-onset
        case_rows_list = []
        for pid, g in sub.groupby("participant_id"):
            s = dict(zip(g["session_id"], g["flag"]))
            if W2 not in s or s[W2] == 1:
                continue
            if W1 in s and s[W1] == 1:
                continue
            if W3 not in s and W4 not in s:
                continue
            if W3 in s and s[W3] == 1:
                case_rows_list.append({"participant_id": pid, "onset": 1})
                continue
            if W4 in s and s[W4] == 1:
                if W3 in s and s[W3] == 0:
                    case_rows_list.append({"participant_id": pid, "onset": 1})

        cases_df = pd.DataFrame(case_rows_list)
        hc_df = pd.DataFrame({"participant_id": list(hc_ids), "onset": 0})
        analytic = pd.concat([hc_df, cases_df], ignore_index=True)

        # Baseline W0 T-score for this scale
        w0_baseline = mh[mh["session_id"] == W1][["participant_id", tscore_col]].dropna()
        w0_baseline = w0_baseline.rename(columns={tscore_col: "baseline_tscore"})
        w0_baseline = w0_baseline.drop_duplicates("participant_id")

        # Merge everything
        analytic = (analytic
            .merge(blups, on="participant_id", how="inner")
            .merge(cbcl_w2, on="participant_id", how="left")
            .merge(sex_df[["participant_id", "is_female"]], on="participant_id", how="left")
            .merge(family, on="participant_id", how="left")
            .merge(demo, on="participant_id", how="left")
            .merge(w0_baseline, on="participant_id", how="left")
            .merge(beh, on="participant_id", how="left"))
        analytic = analytic.dropna(subset=["mesor_blup", "age_yrs", "is_female", "family_id"])

        all_needed = demo_cols + ["baseline_tscore"] + beh_cols + rhythm_cols + ["family_id", "onset"]
        shared = analytic.dropna(subset=all_needed).copy()
        n = len(shared)
        n_case = int(shared["onset"].sum())
        log(f"  Shared row set: n = {n:,}, events = {n_case}")

        if n < 50 or n_case < 10:
            log("  SKIPPED (insufficient sample)")
            continue

        blocks = {
            "M1_demographics": demo_cols,
            "M2_+symptoms": demo_cols + ["baseline_tscore"],
            "M3_+behavior": demo_cols + ["baseline_tscore"] + beh_cols,
            "M4_+rhythm": demo_cols + ["baseline_tscore"] + beh_cols + rhythm_cols,
        }

        fits = {}
        for model_name, cols in blocks.items():
            r = _fit(shared, cols)
            fits[model_name] = r
            fit_rows.append({
                "outcome": label, "model": model_name,
                "n": r["n"], "n_events": r["n_events"],
                "auc": r["auc"], "aic": r["aic"],
            })
            log(f"  {model_name:<24s}  n = {r['n']:,}, events = {r['n_events']}, "
                f"AUC = {r['auc']:.3f}")

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
                "outcome": label,
                "reduced": small_name, "full": big_name,
                "chi2": chi2, "df": dof, "p": p,
                "delta_auc": d_auc,
            })
            log(f"    {small_name} -> {big_name}: "
                f"chi2({dof}) = {chi2:.2f}, p = {p:.3g}, ΔAUC = {d_auc:+.4f}")

        m4_fit = fits["M4_+rhythm"]["fit"]
        log(f"\n  M4 rhythm coefficients (OR per SD):")
        for col in rhythm_cols:
            b = float(m4_fit.params[col])
            ci = m4_fit.conf_int().loc[col].astype(float).tolist()
            r = {
                "outcome": label, "predictor": col,
                "or": float(np.exp(b)),
                "or_lo": float(np.exp(ci[0])),
                "or_hi": float(np.exp(ci[1])),
                "p": float(m4_fit.pvalues[col]),
                "n": fits["M4_+rhythm"]["n"],
                "n_events": fits["M4_+rhythm"]["n_events"],
                "auc_m3": fits["M3_+behavior"]["auc"],
                "auc_m4": fits["M4_+rhythm"]["auc"],
            }
            coef_rows.append(r)
            log(f"    {col:<22s}  OR = {r['or']:.2f} "
                f"[{r['or_lo']:.2f}, {r['or_hi']:.2f}], p = {r['p']:.3g}")

    pd.DataFrame(fit_rows).to_csv(TRAJ_DIR / "dsm_incremental_fit.csv", index=False)
    pd.DataFrame(lrt_rows).to_csv(TRAJ_DIR / "dsm_incremental_lrt.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(TRAJ_DIR / "dsm_incremental_m4_coefficients.csv", index=False)

    log(f"\nWrote results to {TRAJ_DIR}")
    (OUTPUTS_DIR / "25_incremental_all_dsm.log").write_text("\n".join(out_lines))


if __name__ == "__main__":
    main()
