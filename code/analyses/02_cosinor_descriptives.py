"""02 · Cosinor-modelling descriptives for the Results paragraph.

Reproduces every number reported in the "Cosinor modeling — Model
specification" paragraph of the manuscript:

    - Per-participant typical-day cosinor R² distribution
    - BLUP descriptives for mesor / amplitude / acrophase (mean, SD, range,
      circular mean for acrophase)
    - Wave-2 cuff-HR ↔ cosinor-mesor correlation
    - Within-person cohort: n with ≥7 valid daily cosinor fits

Outputs:
    results/tables/cosinor_descriptives.tsv
    results/outputs/02_cosinor_descriptives.log
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import (                          # noqa: E402
    TABLES_DIR, OUTPUTS_DIR, COSINOR_BLUP_W2, WITHIN_PERSON_FEATURES,
    PHYS_OUTCOMES, W2,
)

TABLES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def circular_mean_hours(acr_hours: np.ndarray) -> float:
    theta = 2 * np.pi * acr_hours / 24
    Rx = float(np.cos(theta).mean()); Ry = float(np.sin(theta).mean())
    mean = (np.arctan2(Ry, Rx) / (2 * np.pi)) * 24
    return float(mean % 24)


def circular_sd_hours(acr_hours: np.ndarray) -> float:
    theta = 2 * np.pi * acr_hours / 24
    Rx = float(np.cos(theta).mean()); Ry = float(np.sin(theta).mean())
    R = float(np.sqrt(Rx**2 + Ry**2))
    if R <= 0 or R >= 1:
        return float("nan")
    return float(np.sqrt(-2 * np.log(R)) * 24 / (2 * np.pi))


def main() -> None:
    out_lines: list[str] = []
    def log(msg: str = ""):
        print(msg); out_lines.append(msg)

    # ------------------------------------------------------------------
    # Load BLUPs (Wave-2 typical-day cosinor BLUPs from upstream pipeline)
    # ------------------------------------------------------------------
    blups = pl.read_parquet(COSINOR_BLUP_W2).to_pandas()
    valid = blups.dropna(subset=["mesor_blup", "amplitude_blup",
                                   "acrophase_blup", "r_squared"]).copy()
    n_total = len(valid)
    log("=" * 78)
    log("Cosinor descriptives  ·  Wave-2 typical-day BLUPs")
    log("=" * 78)
    log(f"  n participants with valid BLUPs = {n_total:,}")

    # ------------------------------------------------------------------
    # R² distribution
    # ------------------------------------------------------------------
    r2 = valid["r_squared"].to_numpy()
    log("\n--- Per-participant R² ---")
    log(f"  median = {np.median(r2):.3f}")
    log(f"  mean ± SD = {np.mean(r2):.3f} ± {np.std(r2, ddof=1):.3f}")
    log(f"  IQR = [{np.quantile(r2, .25):.3f}, {np.quantile(r2, .75):.3f}]")
    log(f"  fraction R² > .70 = {(r2 > .70).mean()*100:.1f}%")
    log(f"  fraction R² > .50 = {(r2 > .50).mean()*100:.1f}%")

    # ------------------------------------------------------------------
    # BLUP descriptives
    # ------------------------------------------------------------------
    m  = valid["mesor_blup"].to_numpy()
    a  = valid["amplitude_blup"].to_numpy()
    ac = valid["acrophase_blup"].to_numpy()

    log("\n--- BLUP descriptives ---")
    log(f"  Mesor      M ± SD = {m.mean():.2f} ± {m.std(ddof=1):.2f} bpm   "
        f"range = [{m.min():.2f}, {m.max():.2f}] bpm")
    log(f"  Amplitude  M ± SD = {a.mean():.2f} ± {a.std(ddof=1):.2f} bpm   "
        f"range = [{a.min():.2f}, {a.max():.2f}] bpm")
    log(f"  Acrophase  circular M = {circular_mean_hours(ac):.2f} hr   "
        f"circular SD = {circular_sd_hours(ac):.2f} hr   "
        f"(linear range = [{ac.min():.2f}, {ac.max():.2f}] hr)")

    # ------------------------------------------------------------------
    # Mesor ↔ cuff-HR correlation
    # ------------------------------------------------------------------
    phys = pl.read_parquet(PHYS_OUTCOMES).to_pandas()
    cuff = (phys[(phys["session_id"] == W2)]
              [["participant_id", "bp_hrate_mean"]]
              .dropna(subset=["bp_hrate_mean"])
              .drop_duplicates("participant_id"))
    merged = (valid[["subject_id", "mesor_blup"]]
                 .rename(columns={"subject_id": "participant_id"})
                 .merge(cuff, on="participant_id", how="inner"))
    n_corr = len(merged)
    r_pearson, p_pearson = st.pearsonr(merged["mesor_blup"],
                                         merged["bp_hrate_mean"])
    log("\n--- Mesor ↔ cuff-HR correlation (concurrent Wave-2) ---")
    log(f"  n with both measures = {n_corr:,}")
    log(f"  r = {r_pearson:.3f}, p = {p_pearson:.3g}")

    # ------------------------------------------------------------------
    # Within-person cohort coverage (≥7 valid daily fits)
    # ------------------------------------------------------------------
    feats = pd.read_csv(WITHIN_PERSON_FEATURES)
    has_sd_mesor = feats["SD_daily_mesor"].notna()
    n_with_sd = int(has_sd_mesor.sum())
    pct = 100 * n_with_sd / n_total
    log("\n--- Within-person cohort (≥7 valid daily cosinor fits) ---")
    log(f"  n with ≥7 valid daily fits = {n_with_sd:,} of {n_total:,} "
        f"({pct:.1f}%)")
    log(f"  median per-day R² (across-day median per subject) = "
        f"{feats['median_daily_r2'].median():.3f}")
    log(f"  median usable days per subject = "
        f"{feats['n_days_usable_cosinor'].median():.0f}")

    # Within-person SD distributions
    wp = feats[has_sd_mesor].copy()
    sd_m  = wp["SD_daily_mesor"].to_numpy()
    sd_a  = wp["SD_daily_amplitude"].to_numpy()
    sd_ac = wp["SD_daily_acrophase"].to_numpy()
    log("\n--- Within-person SD distributions ---")
    log(f"  SD daily mesor      M ± SD = {sd_m.mean():.2f} ± {sd_m.std(ddof=1):.2f} bpm   "
        f"median = {np.median(sd_m):.2f}   IQR = "
        f"[{np.quantile(sd_m, .25):.2f}, {np.quantile(sd_m, .75):.2f}]")
    log(f"  SD daily amplitude  M ± SD = {sd_a.mean():.2f} ± {sd_a.std(ddof=1):.2f} bpm   "
        f"median = {np.median(sd_a):.2f}   IQR = "
        f"[{np.quantile(sd_a, .25):.2f}, {np.quantile(sd_a, .75):.2f}]")
    log(f"  SD daily acrophase  M ± SD = {sd_ac.mean():.2f} ± {sd_ac.std(ddof=1):.2f} hr   "
        f"median = {np.median(sd_ac):.2f}   IQR = "
        f"[{np.quantile(sd_ac, .25):.2f}, {np.quantile(sd_ac, .75):.2f}]")

    # ------------------------------------------------------------------
    # Save tabular summary for the manuscript
    # ------------------------------------------------------------------
    rows = [
        ("n_valid_BLUPs", n_total, ""),
        ("R2_median", float(np.median(r2)), ""),
        ("R2_mean", float(np.mean(r2)), ""),
        ("R2_SD", float(np.std(r2, ddof=1)), ""),
        ("R2_IQR_lo", float(np.quantile(r2, .25)), ""),
        ("R2_IQR_hi", float(np.quantile(r2, .75)), ""),
        ("R2_pct_gt_0.70", float((r2 > .70).mean()*100), "%"),
        ("R2_pct_gt_0.50", float((r2 > .50).mean()*100), "%"),
        ("mesor_mean", float(m.mean()), "bpm"),
        ("mesor_SD", float(m.std(ddof=1)), "bpm"),
        ("mesor_min", float(m.min()), "bpm"),
        ("mesor_max", float(m.max()), "bpm"),
        ("amplitude_mean", float(a.mean()), "bpm"),
        ("amplitude_SD", float(a.std(ddof=1)), "bpm"),
        ("amplitude_min", float(a.min()), "bpm"),
        ("amplitude_max", float(a.max()), "bpm"),
        ("acrophase_circular_mean", circular_mean_hours(ac), "hr"),
        ("acrophase_circular_SD",   circular_sd_hours(ac),   "hr"),
        ("acrophase_min", float(ac.min()), "hr"),
        ("acrophase_max", float(ac.max()), "hr"),
        ("mesor_cuff_r", float(r_pearson), ""),
        ("mesor_cuff_n", n_corr, ""),
        ("mesor_cuff_p", float(p_pearson), ""),
        ("within_person_n_ge7days", n_with_sd, ""),
        ("within_person_pct_ge7days", pct, "%"),
        ("SD_daily_mesor_mean", float(sd_m.mean()), "bpm"),
        ("SD_daily_mesor_SD", float(sd_m.std(ddof=1)), "bpm"),
        ("SD_daily_mesor_median", float(np.median(sd_m)), "bpm"),
        ("SD_daily_mesor_IQR_lo", float(np.quantile(sd_m, .25)), "bpm"),
        ("SD_daily_mesor_IQR_hi", float(np.quantile(sd_m, .75)), "bpm"),
        ("SD_daily_amplitude_mean", float(sd_a.mean()), "bpm"),
        ("SD_daily_amplitude_SD", float(sd_a.std(ddof=1)), "bpm"),
        ("SD_daily_amplitude_median", float(np.median(sd_a)), "bpm"),
        ("SD_daily_amplitude_IQR_lo", float(np.quantile(sd_a, .25)), "bpm"),
        ("SD_daily_amplitude_IQR_hi", float(np.quantile(sd_a, .75)), "bpm"),
        ("SD_daily_acrophase_mean", float(sd_ac.mean()), "hr"),
        ("SD_daily_acrophase_SD", float(sd_ac.std(ddof=1)), "hr"),
        ("SD_daily_acrophase_median", float(np.median(sd_ac)), "hr"),
        ("SD_daily_acrophase_IQR_lo", float(np.quantile(sd_ac, .25)), "hr"),
        ("SD_daily_acrophase_IQR_hi", float(np.quantile(sd_ac, .75)), "hr"),
    ]
    pd.DataFrame(rows, columns=["statistic", "value", "unit"]).to_csv(
        TABLES_DIR / "cosinor_descriptives.tsv", sep="\t", index=False)
    log(f"\nWrote {TABLES_DIR / 'cosinor_descriptives.tsv'}")

    (OUTPUTS_DIR / "02_cosinor_descriptives.log").write_text("\n".join(out_lines))
    log(f"Wrote {OUTPUTS_DIR / '02_cosinor_descriptives.log'}")


if __name__ == "__main__":
    main()
