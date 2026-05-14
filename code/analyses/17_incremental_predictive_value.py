"""17 · Incremental predictive value of cosinor parameters beyond Wave-2
   sleep and physical-activity behaviors (2-covariate set).

Tests whether the rhythm signal carries prospective predictive information
above and beyond two mechanistically motivated behavioral covariates:
    1. Total daily sleep duration  (short sleep → less low-night-HR pull
       on the 24-h mean)
    2. Mean daily MVPA (= fairly + very-active minutes; sedentary kids →
       chronically elevated HR)

Three strands:

Section 1 — Between-person primary.
    For each outcome (incident depression, obesity, hypertension), compare
    a behaviors-only model (mean sleep + mean MVPA + age + sex) to a
    behaviors-plus-rhythm model adding one between-person cosinor
    parameter (mesor / amplitude / acrophase).

Section 2 — Within-person stability.
    Scale-matched: behaviors-only uses SD-of-daily sleep duration and
    SD-of-daily MVPA (computed from Box daily files, ≥7 days); rhythm
    predictors are SD-of-daily mesor / amplitude / acrophase.

Section 3 — Comorbidity (conditional prediction).
    For the two estimable cells (Design B), behaviors-only uses BOTH the
    mean and SD versions of sleep + MVPA; rhythm predictors are mean and
    SD-of-daily mesor/amp/acro.

All logistic models fit with family-clustered SEs. Predictors z-scored
within each cell's row set. Same row set across nested models so the LRT
is valid.

Outputs:
    results/sensitivity/incremental_predictive_value.csv
    results/outputs/17_incremental_predictive_value.log
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
from utils.paths import (                              # noqa: E402
    TABLES_DIR, OUTPUTS_DIR, RESULTS_DIR, DERIV, BOX, WITHIN_PERSON_FEATURES,
    W1, W2, W3, W4,
)
from utils.cooccurrence import load_cooccurrence_frame  # noqa: E402

SENS_DIR = RESULTS_DIR / "sensitivity"
SENS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

PER_WAVE_SUMMARY = DERIV / "fitbit_summary" / "per_wave_summary.parquet"
DAILY_ACT = BOX / "abcd-data-release-5.1" / "core" / "novel-technologies" / "nt_y_fitb_act_d.csv"
DAILY_SLP = BOX / "abcd-data-release-5.1" / "core" / "novel-technologies" / "nt_y_fitb_slp_d.csv"
W2_EVENT  = "2_year_follow_up_y_arm_1"
MIN_DAYS  = 7

# Two-covariate set: total daily sleep duration + mean daily MVPA
MEAN_COV = ["sleep_period_min", "mvpa_min"]
SD_COV   = ["sd_sleep_period_min", "sd_mvpa_min"]
BETWEEN_PREDICTORS = [
    ("typical_day_mesor",     "mesor"),
    ("typical_day_amplitude", "amplitude"),
    ("typical_day_acrophase", "acrophase"),
]
WITHIN_PREDICTORS = [
    ("SD_daily_mesor",     "SD daily mesor"),
    ("SD_daily_amplitude", "SD daily amplitude"),
    ("SD_daily_acrophase", "SD daily acrophase"),
]
ALL_RHYTHM_PREDICTORS = BETWEEN_PREDICTORS + WITHIN_PREDICTORS

FRAMES = [
    ("analytic_depression.tsv",   "Depression"),
    ("analytic_obesity.tsv",       "Obesity"),
    ("analytic_hypertension.tsv",  "Hypertension"),
]


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


def _load_mean_covariates() -> pd.DataFrame:
    return (pl.read_parquet(PER_WAVE_SUMMARY)
              .filter(pl.col("session_id") == "ses-02A")
              .select(["participant_id", *MEAN_COV])
              .to_pandas())


def _load_sd_covariates() -> pd.DataFrame:
    """SD-of-daily Wave-2 sleep duration and MVPA (=fairly+very active min)."""
    act = pd.read_csv(DAILY_ACT, low_memory=False,
                        usecols=["src_subject_id", "eventname",
                                  "fit_ss_day_min_gt_600",
                                  "fit_ss_total_fairly_active_min",
                                  "fit_ss_total_very_active_min"])
    act = act[(act["eventname"] == W2_EVENT)
                & (act["fit_ss_day_min_gt_600"] == 1)].copy()
    act["mvpa_daily"] = (act["fit_ss_total_fairly_active_min"].fillna(0)
                            + act["fit_ss_total_very_active_min"].fillna(0))
    act["participant_id"] = act["src_subject_id"].str.replace(
        "NDAR_INV", "sub-", regex=False)
    act_sd = (act.groupby("participant_id")
                  .agg(sd_mvpa_min=("mvpa_daily", "std"),
                        n_act_days=("mvpa_daily", "count"))
                  .reset_index())
    act_sd = act_sd[act_sd["n_act_days"] >= MIN_DAYS]

    slp = pd.read_csv(DAILY_SLP, low_memory=False,
                        usecols=["src_subject_id", "eventname",
                                  "fit_ss_sleepperiod_minutes"])
    slp = slp[slp["eventname"] == W2_EVENT].copy()
    slp["participant_id"] = slp["src_subject_id"].str.replace(
        "NDAR_INV", "sub-", regex=False)
    slp_sd = (slp.groupby("participant_id")
                  .agg(sd_sleep_period_min=("fit_ss_sleepperiod_minutes", "std"),
                        n_slp_nights=("fit_ss_sleepperiod_minutes", "count"))
                  .reset_index())
    slp_sd = slp_sd[slp_sd["n_slp_nights"] >= MIN_DAYS]
    return act_sd.merge(slp_sd, on="participant_id", how="inner")


def _fit(df: pd.DataFrame, predictors: list[str]) -> dict:
    use = df.dropna(subset=predictors + ["age_yrs", "is_female",
                                            "family_id", "onset"]).copy()
    # z-score numeric predictors (skip is_female and age which we keep raw
    # in the model but z-scoring continuous rhythm/behavior predictors)
    for c in predictors:
        if c == "is_female":
            continue
        use[c] = _z(use[c])
    use["age_yrs"] = _z(use["age_yrs"])
    X = sm.add_constant(use[predictors + ["age_yrs", "is_female"]],
                          has_constant="add")
    f = sm.Logit(use["onset"], X).fit(
        disp=0, cov_type="cluster",
        cov_kwds={"groups": use["family_id"]}, maxiter=200)
    p_hat = f.predict(X)
    auc = float(roc_auc_score(use["onset"], p_hat))
    return {"fit": f, "n": int(f.nobs),
            "n_events": int(use["onset"].sum()),
            "auc": auc, "aic": float(f.aic), "llf": float(f.llf),
            "k": int(len(f.params)),
            "use": use, "X": X}


def _lrt(reduced: dict, full: dict) -> tuple[float, int, float]:
    chi2 = 2 * (full["llf"] - reduced["llf"])
    dof = full["k"] - reduced["k"]
    p = 1 - st.chi2.cdf(chi2, df=dof) if dof > 0 else float("nan")
    return float(chi2), int(dof), float(p)


def _or_ci_p(f: sm.Logit, col: str) -> tuple[float, float, float, float]:
    b = float(f.params[col])
    ci = f.conf_int().loc[col].astype(float).tolist()
    return (float(np.exp(b)), float(np.exp(ci[0])), float(np.exp(ci[1])),
            float(f.pvalues[col]))


def _compare(df: pd.DataFrame, behavior_cols: list[str],
              rhythm_col: str, rhythm_label: str,
              extra_info: dict | None = None) -> dict:
    """Fit base (rhythm only) on shared row set + behaviors-only + behaviors+rhythm.
    Same row set across all three for valid LRT and AUC comparisons."""
    rs = df.dropna(subset=behavior_cols + [rhythm_col, "age_yrs",
                                              "is_female", "family_id",
                                              "onset"]).copy()
    if len(rs) < 50 or rs["onset"].sum() < 10:
        return None
    base = _fit(rs, [rhythm_col])
    beh = _fit(rs, behavior_cols)
    full = _fit(rs, behavior_cols + [rhythm_col])
    chi2, dof, lrt_p = _lrt(beh, full)
    or_p, or_lo_p, or_hi_p, p_p = _or_ci_p(base["fit"], rhythm_col)
    or_a, or_lo_a, or_hi_a, p_a = _or_ci_p(full["fit"], rhythm_col)
    rec = {
        "predictor": rhythm_label,
        "n": full["n"], "n_events": full["n_events"],
        "or_primary":  or_p,  "ci_lo_primary":  or_lo_p,
        "ci_hi_primary": or_hi_p, "p_primary":  p_p,
        "or_adjusted": or_a,  "ci_lo_adjusted": or_lo_a,
        "ci_hi_adjusted": or_hi_a, "p_adjusted": p_a,
        "base_auc":      base["auc"], "base_aic": base["aic"],
        "behaviors_auc": beh["auc"], "behaviors_aic": beh["aic"],
        "adjusted_auc":  full["auc"], "adjusted_aic": full["aic"],
        "auc_improvement": full["auc"] - beh["auc"],
        "lrt_chi2": chi2, "lrt_df": dof, "lrt_p": lrt_p,
    }
    if extra_info:
        rec.update(extra_info)
    return rec


def main() -> None:
    out_lines: list[str] = []
    def log(msg: str = ""):
        print(msg); out_lines.append(msg)

    log("=" * 78)
    log("Incremental predictive value of cosinor parameters beyond behavior")
    log("=" * 78)

    cov_mean = _load_mean_covariates()
    cov_sd   = _load_sd_covariates()
    wp = (pd.read_csv(WITHIN_PERSON_FEATURES)
              .rename(columns={"subject_id": "participant_id"})
              [["participant_id", "SD_daily_mesor",
                "SD_daily_amplitude", "SD_daily_acrophase"]])
    log(f"  Mean Wave-2 behavioral covariates available: "
        f"n = {cov_mean.dropna(subset=MEAN_COV).shape[0]:,}")
    log(f"  SD-of-daily Wave-2 covariates (≥{MIN_DAYS} days): "
        f"n = {cov_sd.dropna(subset=SD_COV).shape[0]:,}")

    rows: list[dict] = []

    # ============================================================
    # SECTION 1: Between-person × mean behavioral covariates
    # ============================================================
    log("\n" + "=" * 78)
    log("Section 1. Between-person × mean behavioral covariates")
    log("=" * 78)
    for fname, outcome in FRAMES:
        df = pd.read_csv(TABLES_DIR / fname, sep="\t").rename(columns={
            "mesor_blup":     "typical_day_mesor",
            "amplitude_blup": "typical_day_amplitude",
            "acrophase_blup": "typical_day_acrophase",
        })
        df = df.merge(cov_mean, on="participant_id", how="left")
        log(f"\n  --- {outcome} ---")
        for col, lbl in BETWEEN_PREDICTORS:
            rec = _compare(df, MEAN_COV, col, lbl,
                              extra_info={"strand": "between-person",
                                            "outcome": outcome})
            if rec is None:
                log(f"    {lbl:<22s}  <skipped>")
                continue
            rows.append(rec)
            log(f"    {lbl:<22s}  n = {rec['n']:,}, events = {rec['n_events']}")
            log(f"      OR (primary, no behaviors)  = "
                f"{rec['or_primary']:.2f} "
                f"[{rec['ci_lo_primary']:.2f}, {rec['ci_hi_primary']:.2f}], "
                f"p = {rec['p_primary']:.3g}")
            log(f"      OR (behaviors + rhythm)     = "
                f"{rec['or_adjusted']:.2f} "
                f"[{rec['ci_lo_adjusted']:.2f}, {rec['ci_hi_adjusted']:.2f}], "
                f"p = {rec['p_adjusted']:.3g}")
            log(f"      LRT (full vs behaviors-only): "
                f"χ²({rec['lrt_df']}) = {rec['lrt_chi2']:.2f}, "
                f"p = {rec['lrt_p']:.3g}")
            log(f"      AUC behaviors-only = {rec['behaviors_auc']:.3f}; "
                f"adjusted = {rec['adjusted_auc']:.3f}; "
                f"ΔAUC = {rec['auc_improvement']:+.4f}")

    # ============================================================
    # SECTION 2: Within-person × SD-of-daily behavioral covariates
    # ============================================================
    log("\n" + "=" * 78)
    log("Section 2. Within-person × SD-of-daily behavioral covariates")
    log("=" * 78)
    for fname, outcome in FRAMES:
        df = pd.read_csv(TABLES_DIR / fname, sep="\t")
        df = df.merge(cov_sd, on="participant_id", how="left")
        df = df.merge(wp,     on="participant_id", how="left")
        log(f"\n  --- {outcome} ---")
        for col, lbl in WITHIN_PREDICTORS:
            rec = _compare(df, SD_COV, col, lbl,
                              extra_info={"strand": "within-person",
                                            "outcome": outcome})
            if rec is None:
                log(f"    {lbl:<22s}  <skipped>")
                continue
            rows.append(rec)
            log(f"    {lbl:<22s}  n = {rec['n']:,}, events = {rec['n_events']}")
            log(f"      OR (primary, no behaviors)  = "
                f"{rec['or_primary']:.2f} "
                f"[{rec['ci_lo_primary']:.2f}, {rec['ci_hi_primary']:.2f}], "
                f"p = {rec['p_primary']:.3g}")
            log(f"      OR (behaviors + rhythm)     = "
                f"{rec['or_adjusted']:.2f} "
                f"[{rec['ci_lo_adjusted']:.2f}, {rec['ci_hi_adjusted']:.2f}], "
                f"p = {rec['p_adjusted']:.3g}")
            log(f"      LRT: χ²({rec['lrt_df']}) = {rec['lrt_chi2']:.2f}, "
                f"p = {rec['lrt_p']:.3g}")
            log(f"      AUC behaviors-only = {rec['behaviors_auc']:.3f}; "
                f"adjusted = {rec['adjusted_auc']:.3f}; "
                f"ΔAUC = {rec['auc_improvement']:+.4f}")

    # ============================================================
    # SECTION 3: Comorbidity (conditional prediction)
    # ============================================================
    log("\n" + "=" * 78)
    log("Section 3. Comorbidity × (mean + SD) behavioral covariates")
    log("=" * 78)
    co = load_cooccurrence_frame()
    co = co.rename(columns={
        "age_w2": "age_yrs",
        "cosinor_mesor_w2":     "typical_day_mesor",
        "cosinor_amplitude_w2": "typical_day_amplitude",
        "cosinor_acrophase_w2": "typical_day_acrophase",
        "sd_daily_mesor":       "SD_daily_mesor",
        "sd_daily_amplitude":   "SD_daily_amplitude",
        "sd_daily_acrophase":   "SD_daily_acrophase",
    })
    co = co.merge(cov_mean, on="participant_id", how="left")
    co = co.merge(cov_sd,   on="participant_id", how="left")

    COMORBID_CELLS = [
        ("dep_anchor → obesity onset", "dep_first_wave", "obesity_at_w2",
         "obesity_first_wave", "obesity_obs_w3w4", "obesity"),
        ("obesity_anchor → depression onset", "obesity_first_wave", "dep_at_w2",
         "dep_first_wave", "dep_obs_w3w4", "depression"),
    ]
    BEH_COV = MEAN_COV + SD_COV
    for cell_label, anchor_col, target_w2, target_first, target_obs, _ in COMORBID_CELLS:
        sub = co[(co[anchor_col].isin([W1, W2]))
                  & (co[target_w2] == 0)
                  & (co[target_obs])].copy()
        sub["onset"] = sub[target_first].isin([W3, W4]).astype(int)
        log(f"\n  --- {cell_label} ---")
        log(f"    Pre-covariate-restriction: n = {len(sub):,}, "
            f"events = {int(sub['onset'].sum())}")
        for col, lbl in ALL_RHYTHM_PREDICTORS:
            rec = _compare(sub, BEH_COV, col, lbl,
                              extra_info={"strand": "comorbidity",
                                            "outcome": cell_label})
            if rec is None:
                log(f"    {lbl:<22s}  <skipped (n or events too small)>")
                continue
            rows.append(rec)
            flag = "  <-- UNDERPOWERED" if rec["n_events"] < 30 else ""
            log(f"    {lbl:<22s}  n = {rec['n']:,}, events = "
                f"{rec['n_events']}{flag}")
            log(f"      OR (primary, no behaviors)  = "
                f"{rec['or_primary']:.2f} "
                f"[{rec['ci_lo_primary']:.2f}, {rec['ci_hi_primary']:.2f}], "
                f"p = {rec['p_primary']:.3g}")
            log(f"      OR (behaviors + rhythm)     = "
                f"{rec['or_adjusted']:.2f} "
                f"[{rec['ci_lo_adjusted']:.2f}, {rec['ci_hi_adjusted']:.2f}], "
                f"p = {rec['p_adjusted']:.3g}")
            log(f"      LRT: χ²({rec['lrt_df']}) = {rec['lrt_chi2']:.2f}, "
                f"p = {rec['lrt_p']:.3g}")
            log(f"      AUC behaviors-only = {rec['behaviors_auc']:.3f}; "
                f"adjusted = {rec['adjusted_auc']:.3f}; "
                f"ΔAUC = {rec['auc_improvement']:+.4f}")

    # ----- Save aggregated CSV -----
    out = pd.DataFrame(rows)
    out = out[["strand", "outcome", "predictor", "n", "n_events",
                "or_primary", "ci_lo_primary", "ci_hi_primary", "p_primary",
                "or_adjusted", "ci_lo_adjusted", "ci_hi_adjusted", "p_adjusted",
                "lrt_chi2", "lrt_df", "lrt_p",
                "base_auc", "behaviors_auc", "adjusted_auc", "auc_improvement",
                "base_aic", "behaviors_aic", "adjusted_aic"]]
    out_path = SENS_DIR / "incremental_predictive_value.csv"
    out.to_csv(out_path, index=False)
    log(f"\nWrote {out_path}")

    # ----- Flag summary -----
    log("\n" + "=" * 78)
    log("FLAGS")
    log("=" * 78)

    def _fmt(r):
        sig_or = "*" if r["p_adjusted"] < 0.05 else " "
        sig_lrt = "*" if r["lrt_p"] < 0.05 else " "
        return (f"{r['strand']:<14s} {r['outcome']:<32s} {r['predictor']:<22s}  "
                f"OR_adj = {r['or_adjusted']:.2f} (p {sig_or} = {r['p_adjusted']:.3g})  "
                f"LRT χ² = {r['lrt_chi2']:.2f} (p {sig_lrt} = {r['lrt_p']:.3g})  "
                f"ΔAUC = {r['auc_improvement']:+.4f}")

    log("\n  Cells with OR_adj p<.05 AND LRT p<.05 (strongest evidence):")
    for _, r in out[(out["p_adjusted"] < 0.05) & (out["lrt_p"] < 0.05)].iterrows():
        log(f"    {_fmt(r)}")

    log("\n  Cells with OR_adj p<.05 but LRT p≥.05 (interpretively softer):")
    for _, r in out[(out["p_adjusted"] < 0.05) & (out["lrt_p"] >= 0.05)].iterrows():
        log(f"    {_fmt(r)}")

    log("\n  Cells where primary OR was significant but adjusted is not "
        "(attenuates under behavior):")
    for _, r in out[(out["p_primary"] < 0.05)
                      & (out["p_adjusted"] >= 0.05)].iterrows():
        log(f"    {_fmt(r)}")

    log("\n  Cells with ΔAUC ≥ 0.01 (visible predictive-value gain):")
    for _, r in out[out["auc_improvement"] >= 0.01].iterrows():
        log(f"    {_fmt(r)}")

    log("\n  Comorbidity cells with n_events < 30 after covariate restriction:")
    for _, r in out[(out["strand"] == "comorbidity")
                      & (out["n_events"] < 30)].iterrows():
        log(f"    {r['outcome']:<36s} {r['predictor']:<22s}  events = {r['n_events']}")

    (OUTPUTS_DIR / "17_incremental_predictive_value.log").write_text(
        "\n".join(out_lines))


if __name__ == "__main__":
    main()
