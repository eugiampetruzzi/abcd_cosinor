"""06 · Sleep + activity covariate-adjusted primary models.

Robustness check: re-fit the primary onset models for each outcome
(depression, obesity, hypertension) adding sleep/activity covariates derived
from the same Fitbit data. Two predictor families are adjusted with
matched-scale covariates:

  (1) Between-person rhythm × MEAN sleep/activity (Wave-2 averages):
      Predictors: typical-day mesor, amplitude, acrophase (BLUPs)
      Covariates: mean sleep_period_min, mean waso_min,
                  mean daily_steps, mean mets_avg
      (from per_wave_summary.parquet)

  (2) Within-person rhythm × SD-OF-DAILY sleep/activity (matched scale):
      Predictors: SD daily mesor, SD daily amplitude, SD daily acrophase
      Covariates: SD across days of sleep_period_min, waso_min, total_steps,
                  total_ave_met — computed from the Box daily-level CSVs
                  (`nt_y_fitb_slp_d.csv`, `nt_y_fitb_act_d.csv`) restricted
                  to Wave 2.

The second family uses *within-person* covariate variability because
adjusting a within-person rhythm SD for a between-person mean is a category
error — variability needs variability as its scale-matched control.
Participants need ≥7 valid days for the SD-of-daily covariates (matching
the within-person rhythm-feature criterion).

The vendor-supplied resting heart-rate estimate is excluded as a covariate
because it correlated r ≈ .93 with cosinor mesor (see Supplement).

Both base and adjusted models are fit on the same row set (within each
family) so attenuation reflects the covariates rather than missingness.
Multiple-comparison correction follows the primary convention (no BH-FDR
across the three rhythm parameters within a family); p_fdr equals p_raw.

Outputs:
    results/tables/sleep_activity_covariate_adjusted.tsv        (between-person)
    results/sensitivity/within_person_adjustment.csv            (within-person,
                                                                   SD-of-daily)
    results/outputs/06_sleep_activity_covariates.log
"""
from __future__ import annotations
from pathlib import Path
import sys

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import (                          # noqa: E402
    TABLES_DIR, OUTPUTS_DIR, DERIV, RESULTS_DIR, WITHIN_PERSON_FEATURES, BOX,
)
from utils.modeling import fit_logistic_cluster, fmt_or  # noqa: E402

TABLES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SENS_DIR = RESULTS_DIR / "sensitivity"
SENS_DIR.mkdir(parents=True, exist_ok=True)

PER_WAVE_SUMMARY = DERIV / "fitbit_summary" / "per_wave_summary.parquet"
DAILY_ACT  = BOX / "abcd-data-release-5.1" / "core" / "novel-technologies" / "nt_y_fitb_act_d.csv"
DAILY_SLP  = BOX / "abcd-data-release-5.1" / "core" / "novel-technologies" / "nt_y_fitb_slp_d.csv"
W2_EVENT   = "2_year_follow_up_y_arm_1"
MIN_DAYS   = 7

FRAMES = [
    ("analytic_depression.tsv",   "Depression"),
    ("analytic_obesity.tsv",       "Obesity"),
    ("analytic_hypertension.tsv",  "Hypertension"),
]
BETWEEN = [("typical_day_mesor",     "Mesor"),
           ("typical_day_amplitude", "Amplitude"),
           ("typical_day_acrophase", "Acrophase")]
WITHIN = [("SD_daily_mesor",      "SD daily mesor"),
          ("SD_daily_amplitude",  "SD daily amplitude"),
          ("SD_daily_acrophase",  "SD daily acrophase")]
# Mean-of-daily covariates (between-person scale)
MEAN_COV4 = ["sleep_period_min", "waso_min", "daily_steps", "mets_avg"]
# SD-of-daily covariates (within-person scale, computed below)
SD_COV4 = ["sd_sleep_period_min", "sd_waso_min",
            "sd_daily_steps_b", "sd_daily_mets_b"]


def _load_sd_covariates() -> pd.DataFrame:
    """Compute per-participant Wave-2 SD-of-daily sleep/activity covariates
    from the Box daily-level CSVs."""
    act = pd.read_csv(DAILY_ACT, low_memory=False,
                        usecols=["src_subject_id", "eventname",
                                  "fit_ss_day_min_gt_600",
                                  "fit_ss_total_step",
                                  "fit_ss_total_ave_met"])
    act = act[(act["eventname"] == W2_EVENT)
                & (act["fit_ss_day_min_gt_600"] == 1)].copy()
    act["participant_id"] = act["src_subject_id"].str.replace(
        "NDAR_INV", "sub-", regex=False)
    act_sd = (act.groupby("participant_id")
                  .agg(sd_daily_steps_b=("fit_ss_total_step", "std"),
                        sd_daily_mets_b=("fit_ss_total_ave_met", "std"),
                        n_act_days=("fit_ss_total_step", "count"))
                  .reset_index())
    act_sd = act_sd[act_sd["n_act_days"] >= MIN_DAYS]

    slp = pd.read_csv(DAILY_SLP, low_memory=False,
                        usecols=["src_subject_id", "eventname",
                                  "fit_ss_sleepperiod_minutes",
                                  "fit_ss_wake_minutes"])
    slp = slp[slp["eventname"] == W2_EVENT].copy()
    slp["participant_id"] = slp["src_subject_id"].str.replace(
        "NDAR_INV", "sub-", regex=False)
    slp_sd = (slp.groupby("participant_id")
                  .agg(sd_sleep_period_min=("fit_ss_sleepperiod_minutes", "std"),
                        sd_waso_min=("fit_ss_wake_minutes", "std"),
                        n_slp_nights=("fit_ss_sleepperiod_minutes", "count"))
                  .reset_index())
    slp_sd = slp_sd[slp_sd["n_slp_nights"] >= MIN_DAYS]

    return act_sd.merge(slp_sd, on="participant_id", how="inner")


def main() -> None:
    out_lines: list[str] = []
    def log(msg: str = ""):
        print(msg); out_lines.append(msg)

    # Wave-2 between-person (mean) sleep+activity covariates
    cov_mean = (pl.read_parquet(PER_WAVE_SUMMARY)
                    .filter(pl.col("session_id") == "ses-02A")
                    .select(["participant_id", *MEAN_COV4])
                    .to_pandas())
    # Wave-2 within-person (SD-of-daily) sleep+activity covariates
    cov_sd = _load_sd_covariates()
    # Within-person rhythm SD features
    wp = (pd.read_csv(WITHIN_PERSON_FEATURES)
              .rename(columns={"subject_id": "participant_id"})
              [["participant_id", "SD_daily_mesor",
                "SD_daily_amplitude", "SD_daily_acrophase"]])

    log("=" * 78)
    log("Sleep + activity covariate-adjusted primary onset models")
    log("=" * 78)
    log(f"  Mean Wave-2 covariates available:   "
        f"n = {cov_mean.dropna(subset=MEAN_COV4).shape[0]:,}")
    log(f"  SD-of-daily Wave-2 covariates (≥{MIN_DAYS} days): "
        f"n = {cov_sd.dropna(subset=SD_COV4).shape[0]:,}")
    log(f"  Within-person rhythm SD features: n = {len(wp):,}")

    btw_rows: list[dict] = []
    wp_rows: list[dict] = []

    for fname, label in FRAMES:
        log(f"\n=== {label} ===")
        df = pd.read_csv(TABLES_DIR / fname, sep="\t")
        df = df.rename(columns={"mesor_blup": "typical_day_mesor",
                                  "amplitude_blup": "typical_day_amplitude",
                                  "acrophase_blup": "typical_day_acrophase"})
        df = df.merge(cov_mean, on="participant_id", how="left")
        df = df.merge(cov_sd,   on="participant_id", how="left")
        df = df.merge(wp,       on="participant_id", how="left")

        # ---- (1) Between-person rhythm × MEAN covariates ----
        sub_btw = df.dropna(subset=MEAN_COV4).copy()
        n_b = len(sub_btw); n_cb = int(sub_btw["onset"].sum())
        log(f"  [between-person × mean covariates] same-N row set: "
            f"n = {n_b:,}, cases = {n_cb}")
        for col, plabel in BETWEEN:
            base = fit_logistic_cluster(sub_btw, [col], return_predictor=col)
            adj = fit_logistic_cluster(sub_btw, [col, *MEAN_COV4],
                                          return_predictor=col)
            if base is None or adj is None:
                log(f"    {plabel:<20s}  <skipped>")
                continue
            log(f"    {plabel:<20s}  base {fmt_or(base)}")
            log(f"    {plabel:<20s}  adj  {fmt_or(adj)}")
            btw_rows.append({
                "frame": label, "predictor": col, "predictor_label": plabel,
                "n": base.n, "n_cases": base.n_cases,
                "OR_base": base.OR, "OR_base_lo": base.OR_lo,
                "OR_base_hi": base.OR_hi, "p_base": base.p,
                "OR_adj":  adj.OR,  "OR_adj_lo":  adj.OR_lo,
                "OR_adj_hi":  adj.OR_hi,  "p_adj":  adj.p,
            })

        # ---- (2) Within-person rhythm × SD-OF-DAILY covariates ----
        log(f"  [within-person × SD-of-daily covariates] "
            f"(same-N row set defined per predictor)")
        for col, plabel in WITHIN:
            sub_wp = df.dropna(subset=SD_COV4 + [col]).copy()
            n_w = len(sub_wp); n_cw = int(sub_wp["onset"].sum())
            base = fit_logistic_cluster(sub_wp, [col], return_predictor=col)
            adj = fit_logistic_cluster(sub_wp, [col, *SD_COV4],
                                          return_predictor=col)
            if base is None or adj is None:
                log(f"    {plabel:<20s}  <skipped>")
                continue
            log(f"    {plabel:<20s}  n = {n_w:,}, cases = {n_cw}; "
                f"base {fmt_or(base)} | adj {fmt_or(adj)}")
            wp_rows.append({
                "outcome": label,
                "predictor": col,
                "predictor_label": plabel,
                "or_per_sd_base": base.OR,
                "ci_lo_base": base.OR_lo, "ci_hi_base": base.OR_hi,
                "p_raw_base": base.p, "p_fdr_base": base.p,
                "or_per_sd_adj":  adj.OR,
                "ci_lo_adj": adj.OR_lo, "ci_hi_adj": adj.OR_hi,
                "p_raw_adj":  adj.p, "p_fdr_adj":  adj.p,
                "n": base.n, "n_events": base.n_cases,
            })

    pd.DataFrame(btw_rows).to_csv(
        TABLES_DIR / "sleep_activity_covariate_adjusted.tsv",
        sep="\t", index=False)
    pd.DataFrame(wp_rows).to_csv(
        SENS_DIR / "within_person_adjustment.csv", index=False)
    log(f"\nWrote {TABLES_DIR / 'sleep_activity_covariate_adjusted.tsv'}")
    log(f"Wrote {SENS_DIR / 'within_person_adjustment.csv'}")

    # ---- Side-by-side: base (unadjusted) vs adjusted, within-person ----
    log("\n--- Within-person base vs adjusted side-by-side ---")
    for outcome in ["Depression", "Obesity", "Hypertension"]:
        rows = [r for r in wp_rows if r["outcome"] == outcome]
        if not rows:
            continue
        log(f"\n  {outcome}:")
        for r in rows:
            shift = r["or_per_sd_adj"] - r["or_per_sd_base"]
            sig_flip = ((r["p_raw_base"] < 0.05) !=
                          (r["p_raw_adj"] < 0.05))
            flag = ""
            if abs(shift) > 0.10:
                flag += " [OR shift > 0.10]"
            if sig_flip:
                flag += " [sig flip]"
            log(f"    {r['predictor_label']:<22s}  "
                f"base OR = {r['or_per_sd_base']:.2f} "
                f"[{r['ci_lo_base']:.2f}, {r['ci_hi_base']:.2f}], "
                f"p = {r['p_raw_base']:.3g}  ||  "
                f"adj OR = {r['or_per_sd_adj']:.2f} "
                f"[{r['ci_lo_adj']:.2f}, {r['ci_hi_adj']:.2f}], "
                f"p = {r['p_raw_adj']:.3g}{flag}")

    (OUTPUTS_DIR / "06_sleep_activity_covariates.log").write_text(
        "\n".join(out_lines))


if __name__ == "__main__":
    main()
