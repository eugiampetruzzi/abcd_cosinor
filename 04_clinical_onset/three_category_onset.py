"""Three-category onset: Depression, Anxiety, combined Externalizing, obesity,
hypertension vs the ultra-clean control pool (n=1,188). Externalizing onset =
first T>=65 on any of ADHD/ODD/Conduct. Part A: between- and within-person
models with BH-FDR over 30 tests. Part B: hierarchical M1->M4 increments.
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
from utils.outcomes import load_sex, load_family, load_physical_health
from utils.modeling import fit_logistic_cluster, fmt_or, bh_fdr

TRAJ_DIR = RESULTS_DIR / "trajectory"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEMO_TSV = ONEDRIVE / "Release 6.1" / "Actigraphy_Eu_Outputs" / "subject_demographics.tsv"
PER_WAVE = DERIV / "fitbit_summary" / "per_wave_summary.parquet"

EXT_COMPONENTS = ["cbcl_dsm_adhd_tscore", "cbcl_dsm_opp_tscore", "cbcl_dsm_cond_tscore"]

out_lines: list[str] = []
def log(msg: str = ""):
    print(msg); out_lines.append(msg)


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


def main():
    log("=" * 78)
    log("27 - Three-category framing (Depression / Anxiety / combined Externalizing)")
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

    mh_w0 = mh[mh["session_id"] == W1].drop_duplicates("participant_id")
    phys_w0 = phys[phys["session_id"] == W1].drop_duplicates("participant_id")
    phys_w2 = phys[phys["session_id"] == W2].drop_duplicates("participant_id")

    # Ultra-clean HC pool (identical to script 26)
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

    demo_cols_from_merge = [c for c in demo_df.columns if c != "participant_id"]
    demo_cols = ["age_yrs", "is_female"] + demo_cols_from_merge
    rhythm_cols = ["mesor_blup", "amplitude_blup", "acrophase_blup"]
    beh_cols = ["sleep_period_min", "mvpa_min"]

    def _attach(analytic, w0_base):
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
        return analytic.rename(columns={
            "mesor_blup": "typical_day_mesor",
            "amplitude_blup": "typical_day_amplitude",
            "acrophase_blup": "typical_day_acrophase",
        })

    def _first_onset(flag_by_wave: dict) -> int | None:
        """Return 1 (incident), 0 (control), or None (undefined) given a
        {session_id: 0/1} dict, using the canonical first-onset rule."""
        s = flag_by_wave
        if W2 not in s or s[W2] == 1: return None
        if W1 in s and s[W1] == 1: return None
        if W3 not in s and W4 not in s: return None
        if W3 in s and s[W3] == 1: return 1
        if W4 in s and s[W4] == 1:
            return 1 if (W3 in s and s[W3] == 0) else None
        return None  # cases only; controls come from the ultra-clean pool

    # Single-scale DSM frame (Depression, Anxiety)
    def build_single(tscore_col):
        sub = mh[["participant_id", "session_id", tscore_col]].dropna(
            subset=[tscore_col]).copy()
        sub["flag"] = (sub[tscore_col] >= 65).astype(int)
        sub = sub[sub["participant_id"].isin(cosinor_ids)]
        has_fu = set(sub[sub["session_id"].isin([W3, W4])]["participant_id"])
        hc_ids = ultra_hc & has_fu
        cases = []
        for pid, g in sub.groupby("participant_id"):
            if _first_onset(dict(zip(g["session_id"], g["flag"]))) == 1:
                cases.append(pid)
        analytic = pd.concat([
            pd.DataFrame({"participant_id": list(hc_ids), "onset": 0}),
            pd.DataFrame({"participant_id": cases, "onset": 1})], ignore_index=True)
        w0 = mh_w0[["participant_id", tscore_col]].dropna().rename(
            columns={tscore_col: "baseline_tscore"})
        return _attach(analytic, w0)

    # Combined externalizing frame (ANY of ADHD/ODD/Conduct)
    def build_externalizing():
        sub = mh[["participant_id", "session_id"] + EXT_COMPONENTS].copy()
        sub = sub.dropna(subset=EXT_COMPONENTS, how="all")
        sub["ext_max"] = sub[EXT_COMPONENTS].max(axis=1)
        sub = sub.dropna(subset=["ext_max"])
        sub["flag"] = (sub["ext_max"] >= 65).astype(int)
        sub = sub[sub["participant_id"].isin(cosinor_ids)]
        has_fu = set(sub[sub["session_id"].isin([W3, W4])]["participant_id"])
        hc_ids = ultra_hc & has_fu
        cases = []
        for pid, g in sub.groupby("participant_id"):
            if _first_onset(dict(zip(g["session_id"], g["flag"]))) == 1:
                cases.append(pid)
        analytic = pd.concat([
            pd.DataFrame({"participant_id": list(hc_ids), "onset": 0}),
            pd.DataFrame({"participant_id": cases, "onset": 1})], ignore_index=True)
        w0 = mh_w0[["participant_id"] + EXT_COMPONENTS].copy()
        w0["baseline_tscore"] = w0[EXT_COMPONENTS].max(axis=1)
        w0 = w0[["participant_id", "baseline_tscore"]].dropna()
        return _attach(analytic, w0)

    # Cardiometabolic frame (Obesity, Hypertension)
    def build_cardio(slug):
        if slug == "obesity":
            old = pd.read_csv(TABLES_DIR / "analytic_obesity.tsv", sep="\t")
            w0 = phys_w0[["participant_id", "bmi"]].dropna().rename(
                columns={"bmi": "baseline_tscore"})
        else:
            old = pd.read_csv(TABLES_DIR / "analytic_hypertension.tsv", sep="\t")
            w0 = phys_w2[["participant_id", "bp_sys_mean"]].dropna().rename(
                columns={"bp_sys_mean": "baseline_tscore"})
        case_ids = set(old[old["onset"] == 1]["participant_id"])
        hc_ids = ultra_hc & set(old[old["onset"] == 0]["participant_id"])
        analytic = pd.concat([
            pd.DataFrame({"participant_id": list(hc_ids), "onset": 0}),
            pd.DataFrame({"participant_id": list(case_ids), "onset": 1})], ignore_index=True)
        return _attach(analytic, w0)

    # Assemble the 5 outcomes
    all_outcomes = [
        ("Depression",    build_single("cbcl_dsm_dep_tscore")),
        ("Anxiety",       build_single("cbcl_dsm_anx_tscore")),
        ("Externalizing", build_externalizing()),
        ("Obesity",       build_cardio("obesity")),
        ("Hypertension",  build_cardio("hypertension")),
    ]

    # PART A: between/within onset + FDR(30)
    log("\n" + "=" * 78)
    log("PART A: Between/within-person onset (ultra-clean HC, FDR across 30 tests)")
    log("=" * 78)
    BETWEEN = [("typical_day_mesor", "Mesor"), ("typical_day_amplitude", "Amplitude"),
               ("typical_day_acrophase", "Acrophase")]
    WITHIN = [("SD_daily_mesor", "SD Mesor"), ("SD_daily_amplitude", "SD Amplitude"),
              ("SD_daily_acrophase", "SD Acrophase")]
    rows = []
    for label, analytic in all_outcomes:
        n = len(analytic); n_case = int(analytic["onset"].sum())
        log(f"\n{label}: n={n}, cases={n_case}, HCs={n - n_case}")
        for col, pl_lab in BETWEEN:
            r = fit_logistic_cluster(analytic, [col], return_predictor=col)
            rows.append({"outcome": label, "analysis": "between", "predictor": pl_lab,
                         "n": r.n, "n_cases": r.n_cases, "OR": r.OR,
                         "OR_lo": r.OR_lo, "OR_hi": r.OR_hi, "p": r.p})
            log(f"  {pl_lab:<12s} {fmt_or(r)}")
        for col, pl_lab in WITHIN:
            sub_wp = analytic.dropna(subset=[col]).copy()
            r = fit_logistic_cluster(sub_wp, [col], return_predictor=col)
            rows.append({"outcome": label, "analysis": "within", "predictor": pl_lab,
                         "n": r.n, "n_cases": r.n_cases, "OR": r.OR,
                         "OR_lo": r.OR_lo, "OR_hi": r.OR_hi, "p": r.p})
            log(f"  {pl_lab:<12s} {fmt_or(r)}")
    bw = pd.DataFrame(rows)
    bw["p_fdr"] = bh_fdr(bw["p"].tolist())
    bw.to_csv(TRAJ_DIR / "threecat_between_within.csv", index=False)
    log(f"\n  FDR applied across {len(bw)} tests (5 outcomes x 6 predictors).")

    # PART B: incremental M1->M4
    log("\n\n" + "=" * 78)
    log("PART B: Incremental M1->M4 (ultra-clean HC)")
    log("=" * 78)

    def _fit(df, x_cols):
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
        auc = float(roc_auc_score(use["onset"], f.predict(X)))
        return {"fit": f, "n": int(f.nobs), "n_events": int(use["onset"].sum()),
                "auc": auc, "llf": float(f.llf), "k": int(len(f.params))}

    def _lrt(small, big):
        chi2 = 2 * (big["llf"] - small["llf"]); dof = big["k"] - small["k"]
        return chi2, dof, float(1 - st.chi2.cdf(chi2, df=dof)) if dof > 0 else float("nan")

    rr = ["typical_day_mesor", "typical_day_amplitude", "typical_day_acrophase"]
    fit_rows, lrt_rows, coef_rows = [], [], []
    for label, analytic in all_outcomes:
        bc = "baseline_tscore"
        need = demo_cols + [bc] + beh_cols + rr + ["family_id", "onset"]
        shared = analytic.dropna(subset=need).copy()
        n = len(shared); n_case = int(shared["onset"].sum())
        log(f"\n{label}: shared n={n}, events={n_case}")
        if n < 50 or n_case < 10:
            log("  SKIPPED"); continue
        blocks = {
            "M1_demographics": demo_cols,
            "M2_+symptoms": demo_cols + [bc],
            "M3_+behavior": demo_cols + [bc] + beh_cols,
            "M4_+rhythm": demo_cols + [bc] + beh_cols + rr,
        }
        fits = {}
        for mn, cols in blocks.items():
            r = _fit(shared, cols); fits[mn] = r
            fit_rows.append({"outcome": label, "model": mn, "n": r["n"],
                             "n_events": r["n_events"], "auc": r["auc"]})
            log(f"  {mn:<24s} n={r['n']}, events={r['n_events']}, AUC={r['auc']:.3f}")
        for sm_n, bg_n in [("M1_demographics", "M2_+symptoms"),
                           ("M2_+symptoms", "M3_+behavior"),
                           ("M3_+behavior", "M4_+rhythm")]:
            chi2, dof, p = _lrt(fits[sm_n], fits[bg_n])
            d_auc = fits[bg_n]["auc"] - fits[sm_n]["auc"]
            lrt_rows.append({"outcome": label, "reduced": sm_n, "full": bg_n,
                             "chi2": chi2, "df": dof, "p": p, "delta_auc": d_auc})
            log(f"    {sm_n} -> {bg_n}: chi2({dof})={chi2:.2f}, p={p:.3g}, ΔAUC={d_auc:+.4f}")
        m4 = fits["M4_+rhythm"]["fit"]
        log("  M4 rhythm ORs:")
        for col in rr:
            b = float(m4.params[col]); ci = m4.conf_int().loc[col].astype(float).tolist()
            r = {"outcome": label, "predictor": col, "or": float(np.exp(b)),
                 "or_lo": float(np.exp(ci[0])), "or_hi": float(np.exp(ci[1])),
                 "p": float(m4.pvalues[col]), "n": fits["M4_+rhythm"]["n"],
                 "n_events": fits["M4_+rhythm"]["n_events"],
                 "auc_m3": fits["M3_+behavior"]["auc"], "auc_m4": fits["M4_+rhythm"]["auc"]}
            coef_rows.append(r)
            sig = "***" if r["p"] < .001 else "**" if r["p"] < .01 else "*" if r["p"] < .05 else ""
            log(f"    {col:<24s} OR={r['or']:.2f} [{r['or_lo']:.2f}, {r['or_hi']:.2f}] p={r['p']:.3g} {sig}")

    pd.DataFrame(fit_rows).to_csv(TRAJ_DIR / "threecat_incremental_fit.csv", index=False)
    pd.DataFrame(lrt_rows).to_csv(TRAJ_DIR / "threecat_incremental_lrt.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(TRAJ_DIR / "threecat_m4_coefficients.csv", index=False)

    (OUTPUTS_DIR / "27_three_category.log").write_text("\n".join(out_lines))
    log("\nDone.")


if __name__ == "__main__":
    main()
