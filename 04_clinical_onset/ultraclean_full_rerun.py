"""Full 8-outcome onset rerun against the ultra-clean control pool (below
threshold on all 6 DSM scales + obesity + hypertension at every observed wave).
Part A: between- and within-person models. Part B: hierarchical M1->M4. The
paper's three-category framing is the reduced version (three_category_onset.py).
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
from utils.paths import *
from utils.outcomes import load_sex, load_family, load_mental_health, load_physical_health
from utils.modeling import fit_logistic_cluster, fmt_or, bh_fdr

TRAJ_DIR = RESULTS_DIR / "trajectory"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEMO_TSV = ONEDRIVE / "Release 6.1" / "Actigraphy_Eu_Outputs" / "subject_demographics.tsv"
PER_WAVE = DERIV / "fitbit_summary" / "per_wave_summary.parquet"

out_lines: list[str] = []
def log(msg: str = ""):
    print(msg); out_lines.append(msg)


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


def main():
    log("=" * 78)
    log("26 - Full rerun — ultra-clean HC pool")
    log("=" * 78)

    # Load data
    mh = pd.read_parquet(MH_OUTCOMES)
    sex_df = load_sex()
    family = load_family()
    phys = load_physical_health(sex=sex_df)

    blups = (pl.read_parquet(COSINOR_BLUP_W2)
                .filter(pl.col("r_squared").is_not_null())
                .select(["subject_id", "mesor_blup", "amplitude_blup", "acrophase_blup"])
                .rename({"subject_id": "participant_id"})
                .to_pandas())
    cosinor_ids = set(blups["participant_id"])

    wp_feats = (pd.read_csv(WITHIN_PERSON_FEATURES)
                  .rename(columns={"subject_id": "participant_id"}))

    cbcl_w2 = (mh[mh["session_id"] == W2][["participant_id", "cbcl_age"]]
                 .drop_duplicates("participant_id")
                 .rename(columns={"cbcl_age": "age_yrs"})
                 .dropna(subset=["age_yrs"]))

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
    demo_df = demo[demo_keep]

    beh = (pl.read_parquet(PER_WAVE)
             .filter(pl.col("session_id") == "ses-02A")
             .select(["participant_id", "sleep_period_min", "mvpa_min"])
             .to_pandas()
             .dropna())

    # Baseline symptoms (W0)
    mh_w0 = mh[mh["session_id"] == W1].drop_duplicates("participant_id")
    phys_w0 = phys[phys["session_id"] == W1].drop_duplicates("participant_id")
    phys_w2 = phys[phys["session_id"] == W2].drop_duplicates("participant_id")

    # Build ultra-clean HC pool
    analytic_dep_old = pd.read_csv(TABLES_DIR / "analytic_depression.tsv", sep="\t")
    base_hc = set(analytic_dep_old[analytic_dep_old["onset"] == 0]["participant_id"])

    dsm_cols = [
        "cbcl_dsm_dep_tscore", "cbcl_dsm_anx_tscore", "cbcl_dsm_adhd_tscore",
        "cbcl_dsm_somat_tscore", "cbcl_dsm_cond_tscore", "cbcl_dsm_opp_tscore",
    ]
    ever_elevated = set()
    for col in dsm_cols:
        sub = mh[["participant_id", col]].dropna()
        ever_elevated |= set(sub[sub[col] >= 65]["participant_id"])

    ultra_hc = base_hc - ever_elevated
    log(f"Ultra-clean HC pool: {len(ultra_hc)}")
    log(f"  (base HC: {len(base_hc)}, removed {len(base_hc) - len(ultra_hc)} "
        f"with any DSM elevation)")

    # Define all outcomes
    DSM_SCALES = [
        ("cbcl_dsm_dep_tscore",   "Depression"),
        ("cbcl_dsm_anx_tscore",   "Anxiety"),
        ("cbcl_dsm_adhd_tscore",  "ADHD"),
        ("cbcl_dsm_somat_tscore", "Somatic"),
        ("cbcl_dsm_cond_tscore",  "Conduct"),
        ("cbcl_dsm_opp_tscore",   "ODD"),
    ]

    demo_cols_from_merge = [c for c in demo_df.columns if c != "participant_id"]
    demo_cols = ["age_yrs", "is_female"] + demo_cols_from_merge
    rhythm_cols = ["mesor_blup", "amplitude_blup", "acrophase_blup"]
    beh_cols = ["sleep_period_min", "mvpa_min"]

    # Helper: build DSM analytic frame
    def build_dsm_frame(tscore_col):
        sub = mh[["participant_id", "session_id", tscore_col]].dropna(
            subset=[tscore_col]).copy()
        sub["flag"] = (sub[tscore_col] >= 65).astype(int)
        sub = sub[sub["participant_id"].isin(cosinor_ids)]

        has_followup = set(sub[sub["session_id"].isin([W3, W4])]["participant_id"])
        hc_ids = ultra_hc & has_followup

        case_rows = []
        for pid, g in sub.groupby("participant_id"):
            s = dict(zip(g["session_id"], g["flag"]))
            if W2 not in s or s[W2] == 1: continue
            if W1 in s and s[W1] == 1: continue
            if W3 not in s and W4 not in s: continue
            if W3 in s and s[W3] == 1:
                case_rows.append({"participant_id": pid, "onset": 1}); continue
            if W4 in s and s[W4] == 1:
                if W3 in s and s[W3] == 0:
                    case_rows.append({"participant_id": pid, "onset": 1})

        cases_df = pd.DataFrame(case_rows)
        hc_df = pd.DataFrame({"participant_id": list(hc_ids), "onset": 0})
        analytic = pd.concat([hc_df, cases_df], ignore_index=True)

        w0_base = mh_w0[["participant_id", tscore_col]].dropna()
        w0_base = w0_base.rename(columns={tscore_col: "baseline_tscore"})

        analytic = (analytic
            .merge(blups, on="participant_id", how="inner")
            .merge(cbcl_w2, on="participant_id", how="left")
            .merge(sex_df[["participant_id", "is_female"]], on="participant_id", how="left")
            .merge(family, on="participant_id", how="left")
            .merge(demo_df, on="participant_id", how="left")
            .merge(w0_base, on="participant_id", how="left")
            .merge(beh, on="participant_id", how="left")
            .merge(wp_feats[["participant_id", "SD_daily_mesor",
                              "SD_daily_amplitude", "SD_daily_acrophase"]],
                   on="participant_id", how="left"))
        analytic = analytic.dropna(subset=["mesor_blup", "age_yrs", "is_female", "family_id"])
        analytic = analytic.rename(columns={
            "mesor_blup": "typical_day_mesor",
            "amplitude_blup": "typical_day_amplitude",
            "acrophase_blup": "typical_day_acrophase",
        })
        return analytic

    # Helper: build cardiometabolic frame
    def build_cardio_frame(slug):
        if slug == "obesity":
            analytic_old = pd.read_csv(TABLES_DIR / "analytic_obesity.tsv", sep="\t")
            baseline_col = "baseline_bmi"
            w0_base = phys_w0[["participant_id", "bmi"]].dropna()
            w0_base = w0_base.rename(columns={"bmi": "baseline_bmi"})
        else:
            analytic_old = pd.read_csv(TABLES_DIR / "analytic_hypertension.tsv", sep="\t")
            baseline_col = "baseline_sbp"
            w0_base = phys_w2[["participant_id", "bp_sys_mean"]].dropna()
            w0_base = w0_base.rename(columns={"bp_sys_mean": "baseline_sbp"})

        case_ids = set(analytic_old[analytic_old["onset"] == 1]["participant_id"])
        hc_ids = ultra_hc & set(analytic_old[analytic_old["onset"] == 0]["participant_id"])

        hc_df = pd.DataFrame({"participant_id": list(hc_ids), "onset": 0})
        cases_df = pd.DataFrame({"participant_id": list(case_ids), "onset": 1})
        analytic = pd.concat([hc_df, cases_df], ignore_index=True)

        analytic = (analytic
            .merge(blups, on="participant_id", how="inner")
            .merge(cbcl_w2, on="participant_id", how="left")
            .merge(sex_df[["participant_id", "is_female"]], on="participant_id", how="left")
            .merge(family, on="participant_id", how="left")
            .merge(demo_df, on="participant_id", how="left")
            .merge(w0_base, on="participant_id", how="left")
            .merge(beh, on="participant_id", how="left")
            .merge(wp_feats[["participant_id", "SD_daily_mesor",
                              "SD_daily_amplitude", "SD_daily_acrophase"]],
                   on="participant_id", how="left"))
        analytic = analytic.dropna(subset=["mesor_blup", "age_yrs", "is_female", "family_id"])
        analytic["baseline_tscore"] = analytic.get(baseline_col, np.nan)
        analytic = analytic.rename(columns={
            "mesor_blup": "typical_day_mesor",
            "amplitude_blup": "typical_day_amplitude",
            "acrophase_blup": "typical_day_acrophase",
        })
        return analytic, baseline_col

    # Fit helpers
    def _fit_logistic(df, x_cols):
        use = df.dropna(subset=x_cols + ["family_id", "onset"]).copy()
        for c in x_cols:
            if use[c].nunique() > 2:
                use[c] = _z(use[c])
        X = sm.add_constant(use[x_cols], has_constant="add")
        try:
            f = sm.Logit(use["onset"].astype(float), X).fit(
                disp=0, cov_type="cluster",
                cov_kwds={"groups": use["family_id"]}, maxiter=300)
        except Exception:
            f = sm.Logit(use["onset"].astype(float), X).fit(
                disp=0, method="bfgs", cov_type="cluster",
                cov_kwds={"groups": use["family_id"]}, maxiter=500)
        p_hat = f.predict(X)
        auc = float(roc_auc_score(use["onset"], p_hat))
        return {"fit": f, "n": int(f.nobs), "n_events": int(use["onset"].sum()),
                "auc": auc, "aic": float(f.aic), "llf": float(f.llf),
                "k": int(len(f.params))}

    def _lrt(small, big):
        chi2 = 2 * (big["llf"] - small["llf"])
        dof = big["k"] - small["k"]
        p = float(1 - st.chi2.cdf(chi2, df=dof)) if dof > 0 else float("nan")
        return chi2, dof, p

    # PART A: Between/within-person onset
    log("\n" + "=" * 78)
    log("PART A: Between-person and within-person onset (ultra-clean HC)")
    log("=" * 78)

    btw_within_rows = []

    BETWEEN = [
        ("typical_day_mesor",     "Mesor"),
        ("typical_day_amplitude", "Amplitude"),
        ("typical_day_acrophase", "Acrophase"),
    ]
    WITHIN = [
        ("SD_daily_mesor",     "SD Mesor"),
        ("SD_daily_amplitude", "SD Amplitude"),
        ("SD_daily_acrophase", "SD Acrophase"),
    ]

    all_outcomes = []
    for tscore_col, label in DSM_SCALES:
        df = build_dsm_frame(tscore_col)
        all_outcomes.append((label, df))

    for slug, label in [("obesity", "Obesity"), ("hypertension", "Hypertension")]:
        df, _ = build_cardio_frame(slug)
        all_outcomes.append((label, df))

    for label, analytic in all_outcomes:
        n = len(analytic); n_case = int(analytic["onset"].sum())
        log(f"\n{label}: n={n}, cases={n_case}, HCs={n - n_case}")

        for col, pred_label in BETWEEN:
            r = fit_logistic_cluster(analytic, [col], return_predictor=col)
            if r is None: continue
            btw_within_rows.append({
                "outcome": label, "analysis": "between", "predictor": pred_label,
                "n": r.n, "n_cases": r.n_cases,
                "OR": r.OR, "OR_lo": r.OR_lo, "OR_hi": r.OR_hi, "p": r.p})
            sig = "***" if r.p < .001 else "**" if r.p < .01 else "*" if r.p < .05 else ""
            log(f"  {pred_label:<12s}  {fmt_or(r)} {sig}")

        for col, pred_label in WITHIN:
            sub_wp = analytic.dropna(subset=[col]).copy()
            if sub_wp["onset"].sum() < 20: continue
            r = fit_logistic_cluster(sub_wp, [col], return_predictor=col)
            if r is None: continue
            btw_within_rows.append({
                "outcome": label, "analysis": "within", "predictor": pred_label,
                "n": r.n, "n_cases": r.n_cases,
                "OR": r.OR, "OR_lo": r.OR_lo, "OR_hi": r.OR_hi, "p": r.p})
            sig = "***" if r.p < .001 else "**" if r.p < .01 else "*" if r.p < .05 else ""
            log(f"  {pred_label:<12s}  {fmt_or(r)} {sig}")

    bw_df = pd.DataFrame(btw_within_rows)
    bw_df["p_fdr"] = bh_fdr(bw_df["p"].tolist())
    bw_df.to_csv(TRAJ_DIR / "ultraclean_between_within.csv", index=False)

    # PART B: Incremental M1->M4
    log("\n\n" + "=" * 78)
    log("PART B: Incremental M1→M4 (ultra-clean HC)")
    log("=" * 78)

    fit_rows = []
    lrt_rows = []
    coef_rows = []

    for label, analytic in all_outcomes:
        # Determine baseline column
        if label in ("Obesity",):
            base_col = "baseline_bmi" if "baseline_bmi" in analytic.columns else "baseline_tscore"
        elif label in ("Hypertension",):
            base_col = "baseline_sbp" if "baseline_sbp" in analytic.columns else "baseline_tscore"
        else:
            base_col = "baseline_tscore"

        if base_col not in analytic.columns:
            analytic[base_col] = np.nan

        all_needed = demo_cols + [base_col] + beh_cols + \
                     ["typical_day_mesor", "typical_day_amplitude", "typical_day_acrophase"] + \
                     ["family_id", "onset"]
        shared = analytic.dropna(subset=all_needed).copy()
        n = len(shared); n_case = int(shared["onset"].sum())
        log(f"\n{label}: shared n={n}, events={n_case}")

        if n < 50 or n_case < 10:
            log("  SKIPPED")
            continue

        rhythm_rename = ["typical_day_mesor", "typical_day_amplitude", "typical_day_acrophase"]
        blocks = {
            "M1_demographics": demo_cols,
            "M2_+symptoms": demo_cols + [base_col],
            "M3_+behavior": demo_cols + [base_col] + beh_cols,
            "M4_+rhythm": demo_cols + [base_col] + beh_cols + rhythm_rename,
        }

        fits = {}
        for model_name, cols in blocks.items():
            r = _fit_logistic(shared, cols)
            fits[model_name] = r
            fit_rows.append({
                "outcome": label, "model": model_name,
                "n": r["n"], "n_events": r["n_events"],
                "auc": r["auc"], "aic": r["aic"],
            })
            log(f"  {model_name:<24s}  n={r['n']}, events={r['n_events']}, AUC={r['auc']:.3f}")

        pairs = [
            ("M1_demographics", "M2_+symptoms"),
            ("M2_+symptoms", "M3_+behavior"),
            ("M3_+behavior", "M4_+rhythm"),
        ]
        for small_name, big_name in pairs:
            chi2, dof, p = _lrt(fits[small_name], fits[big_name])
            d_auc = fits[big_name]["auc"] - fits[small_name]["auc"]
            lrt_rows.append({
                "outcome": label, "reduced": small_name, "full": big_name,
                "chi2": chi2, "df": dof, "p": p, "delta_auc": d_auc,
            })
            log(f"    {small_name} -> {big_name}: chi2({dof})={chi2:.2f}, "
                f"p={p:.3g}, ΔAUC={d_auc:+.4f}")

        m4_fit = fits["M4_+rhythm"]["fit"]
        log(f"  M4 rhythm ORs:")
        for col in rhythm_rename:
            b = float(m4_fit.params[col])
            ci = m4_fit.conf_int().loc[col].astype(float).tolist()
            r = {
                "outcome": label, "predictor": col,
                "or": float(np.exp(b)), "or_lo": float(np.exp(ci[0])),
                "or_hi": float(np.exp(ci[1])), "p": float(m4_fit.pvalues[col]),
                "n": fits["M4_+rhythm"]["n"], "n_events": fits["M4_+rhythm"]["n_events"],
                "auc_m3": fits["M3_+behavior"]["auc"],
                "auc_m4": fits["M4_+rhythm"]["auc"],
            }
            coef_rows.append(r)
            sig = "***" if r["p"] < .001 else "**" if r["p"] < .01 else "*" if r["p"] < .05 else ""
            log(f"    {col:<28s}  OR={r['or']:.2f} [{r['or_lo']:.2f}, {r['or_hi']:.2f}] "
                f"p={r['p']:.3g} {sig}")

    pd.DataFrame(fit_rows).to_csv(TRAJ_DIR / "ultraclean_incremental_fit.csv", index=False)
    pd.DataFrame(lrt_rows).to_csv(TRAJ_DIR / "ultraclean_incremental_lrt.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(TRAJ_DIR / "ultraclean_incremental_m4_coefficients.csv", index=False)

    # Sample summary
    log("\n\n" + "=" * 78)
    log("SAMPLE SUMMARY")
    log("=" * 78)
    all_pred_ids = set()
    all_case_ids = set()
    for label, analytic in all_outcomes:
        cases = set(analytic[analytic["onset"] == 1]["participant_id"])
        all_pred_ids |= set(analytic["participant_id"])
        all_case_ids |= cases
        log(f"  {label:<15s}  n={len(analytic)}, cases={int(analytic['onset'].sum())}, "
            f"HCs={len(analytic) - int(analytic['onset'].sum())}")
    log(f"\n  Prediction union: {len(all_pred_ids)}")
    log(f"  Ultra-clean HCs: {len(ultra_hc)}")
    log(f"  Total unique cases: {len(all_case_ids)}")

    (OUTPUTS_DIR / "26_full_rerun_ultraclean.log").write_text("\n".join(out_lines))
    log(f"\nDone.")


if __name__ == "__main__":
    main()
