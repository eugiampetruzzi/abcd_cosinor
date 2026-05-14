"""18c · Within-person SD-matched adjustment — three-signal version.

Mirrors the three non-HR signals in the between-person joint cosinor
(script 18) by decomposing the activity covariate set into steps and METs
and replacing daily sleep duration with daily proportion-of-wear-time-
asleep:

    sd_daily_steps        — SD across valid days of daily total step count
                            (sum of minute-level steps on wear-flagged minutes)
    sd_daily_mets         — SD across valid days of daily mean METs / minute
    sd_daily_prop_asleep  — SD across valid days of daily proportion of
                            wear-flagged minutes labeled asleep
                            (Fitbit Slp1m value ∈ {1, 2})

Parallel to script 06b. Differences:
    - activity is split into steps + METs (was bundled MVPA)
    - sleep uses wear-time-denominated proportion (was raw sleep_period_min)
    - valid-day filter: dairc 4-of-4 quadrant rule (matches script 18 exactly;
      06b uses the looser ABCD ≥600-min rule)

Phase A: parallel per-participant aggregation from nonwear masks + Slp1m TSVs.
Phase B: load analytic_*.tsv (super-healthy) + within-person rhythm SD
         features; fit base (rhythm only) and adjusted (rhythm + 3 behavioral
         SDs) on the same row set. Frame as **incremental predictive value**
         (sleep / activity day-to-day variability is plausibly upstream of
         day-to-day HR rhythm variability — clean confounder interpretation
         not available; the question is whether the rhythm SD carries
         predictive information beyond the behavioral SDs).

Outputs:
    dairc/derivatives/within_person_sd_matched/per_participant.parquet  (cache)
    results/sensitivity/within_person_sd_matched_3signal.tsv
    results/sensitivity/within_person_sd_matched_3signal.md
    results/outputs/18c_within_person_sd_matched.log
"""
from __future__ import annotations
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from scipy import stats as st
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import (                          # noqa: E402
    TABLES_DIR, OUTPUTS_DIR, RESULTS_DIR, DERIV, QC, WITHIN_PERSON_FEATURES,
)

W2 = "ses-02A"
MIN_DAYS = 7
WORKERS = 8

NWM_DIR  = DERIV / "nonwear_masks"
RAW_DIR  = Path("/Users/eu/Desktop/dairc/rawdata")

CACHE_DIR = DERIV / "within_person_sd_matched"
CACHE_FP  = CACHE_DIR / "per_participant.parquet"
SENS_DIR  = RESULTS_DIR / "sensitivity"
SENS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Phase A — per-subject daily aggregation, then participant SD across days
# ---------------------------------------------------------------------------

def _per_subject(args: tuple[str, list]) -> dict | None:
    subject_id, valid_dates = args
    fp = NWM_DIR / f"{subject_id}_{W2}_wearmask.parquet"
    if not fp.exists() or not valid_dates:
        return None
    from datetime import date as _date
    valid_set = [_date.fromisoformat(d) if isinstance(d, str) else d
                   for d in valid_dates]
    df = (pl.read_parquet(fp, columns=["timestamp", "steps", "mets", "wear_flag"])
            .filter(pl.col("wear_flag") == 1)
            .with_columns(pl.col("timestamp").dt.date().alias("date"))
            .filter(pl.col("date").is_in(valid_set)))
    if df.height == 0:
        return None

    # Join Slp1m (asleep = value ∈ {1, 2})
    slp_fp = RAW_DIR / subject_id / W2 / "beh" / f"{subject_id}_{W2}_task-fitbSlp1m_beh.tsv"
    if slp_fp.exists():
        try:
            slp = pl.read_csv(slp_fp, separator="\t",
                                 schema_overrides={"Wear_Time": pl.Datetime,
                                                     "value": pl.Int8},
                                 ignore_errors=True)
            if (slp.height > 0
                    and "Wear_Time" in slp.columns
                    and "value" in slp.columns):
                slp = (slp.rename({"Wear_Time": "timestamp"})
                          .with_columns(pl.col("value").is_in([1, 2])
                                          .cast(pl.Int8).alias("asleep"))
                          .select(["timestamp", "asleep"]))
                df = df.join(slp, on="timestamp", how="left").with_columns(
                    pl.col("asleep").fill_null(0).alias("asleep"))
            else:
                df = df.with_columns(pl.lit(0).cast(pl.Int8).alias("asleep"))
        except Exception:
            df = df.with_columns(pl.lit(0).cast(pl.Int8).alias("asleep"))
    else:
        df = df.with_columns(pl.lit(0).cast(pl.Int8).alias("asleep"))

    daily = (df.group_by("date")
               .agg([pl.col("steps").sum().alias("steps_total"),
                     pl.col("mets").mean().alias("mets_mean"),
                     pl.col("asleep").cast(pl.Float64).mean()
                       .alias("prop_asleep"),
                     pl.col("wear_flag").len().alias("wear_min")])
               .sort("date"))
    if daily.height < MIN_DAYS:
        return None
    steps = np.asarray(daily["steps_total"].to_list(), dtype=float)
    mets  = np.asarray(daily["mets_mean"].to_list(),  dtype=float)
    prop  = np.asarray(daily["prop_asleep"].to_list(), dtype=float)
    wear  = np.asarray(daily["wear_min"].to_list(),   dtype=float)
    return dict(
        subject_id=subject_id,
        n_days=int(daily.height),
        sd_daily_steps=float(np.nanstd(steps, ddof=1))
            if np.sum(np.isfinite(steps)) > 1 else float("nan"),
        sd_daily_mets=float(np.nanstd(mets, ddof=1))
            if np.sum(np.isfinite(mets)) > 1 else float("nan"),
        sd_daily_prop_asleep=float(np.nanstd(prop, ddof=1))
            if np.sum(np.isfinite(prop)) > 1 else float("nan"),
        mean_daily_wear_min=float(np.nanmean(wear))
            if np.sum(np.isfinite(wear)) > 0 else float("nan"),
    )


def build_sd_cache() -> pl.DataFrame:
    if CACHE_FP.exists():
        print(f"  reusing cached {CACHE_FP}")
        return pl.read_parquet(CACHE_FP)
    print(f"  computing per-subject SD-of-daily behavioral covariates …")
    sess = pl.read_csv(QC / "stage1_full_cohort_sessions.tsv", separator="\t")
    days = pl.read_csv(QC / "stage1_full_cohort_days.tsv", separator="\t",
                          try_parse_dates=True)
    pri  = sess.filter((pl.col("session_id") == W2)
                          & pl.col("passes_4of4_min3days"))
    valid = (days.filter((pl.col("session_id") == W2)
                            & pl.col("is_valid_day_4of4"))
                    .select(["subject_id", "date"]).to_pandas())
    valid["date_str"] = valid["date"].astype(str)
    lookup = valid.groupby("subject_id")["date_str"].apply(list).to_dict()
    work = [(r["subject_id"], lookup.get(r["subject_id"], []))
              for r in pri.to_dicts()]
    print(f"    primary sessions: {len(work):,}")

    rows: list[dict] = []
    t0 = time.time(); n_done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for fut in as_completed(ex.submit(_per_subject, w) for w in work):
            res = fut.result()
            n_done += 1
            if res is not None:
                rows.append(res)
            if n_done % 1000 == 0:
                print(f"      [{n_done:>5d}/{len(work)}]  "
                      f"{n_done / (time.time() - t0):.1f}/s")
    el = time.time() - t0
    print(f"    done: {n_done} processed, {len(rows)} retained "
          f"(≥{MIN_DAYS} valid days)  in {el:.1f}s")
    out = pl.DataFrame(rows).sort("subject_id")
    out.write_parquet(CACHE_FP, compression="zstd")
    print(f"    wrote {CACHE_FP}")
    return out


# ---------------------------------------------------------------------------
# Phase B — base + adjusted within-person models
# ---------------------------------------------------------------------------

NEW_SD_COV = ["sd_daily_steps", "sd_daily_mets", "sd_daily_prop_asleep"]
WITHIN = [("SD_daily_mesor",     "SD daily mesor"),
            ("SD_daily_amplitude", "SD daily amplitude"),
            ("SD_daily_acrophase", "SD daily acrophase")]
FRAMES = [("analytic_depression.tsv",   "Depression"),
            ("analytic_obesity.tsv",      "Obesity"),
            ("analytic_hypertension.tsv", "Hypertension")]


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def fit_model(df: pd.DataFrame, x_cols: list[str]) -> dict:
    use = df.dropna(subset=["onset", *x_cols,
                              "age_yrs", "is_female", "family_id"]).copy()
    use["onset"] = use["onset"].astype(int)
    Xz = use[x_cols + ["age_yrs", "is_female"]].copy()
    for c in x_cols:
        Xz[c] = _zscore(Xz[c])
    X = sm.add_constant(Xz, has_constant="add")
    f = sm.Logit(use["onset"], X).fit(
        disp=0, cov_type="cluster",
        cov_kwds={"groups": use["family_id"]}, maxiter=200)
    yhat = f.predict(X)
    return {
        "fit": f, "X": X, "use": use,
        "n":      int(f.nobs),
        "n_cases": int(use["onset"].sum()),
        "loglik": float(f.llf),
        "aic":    float(f.aic),
        "auc":    float(roc_auc_score(use["onset"], yhat)),
        "k":      len(x_cols),
    }


def main() -> None:
    out_lines: list[str] = []
    def log(msg: str = ""):
        print(msg); out_lines.append(msg)

    log("=" * 78)
    log("Within-person SD-matched adjustment, three-signal version")
    log("  (mirrors script 18 by decomposing activity into steps + METs and")
    log("   using daily proportion-of-wear-time-asleep instead of duration)")
    log("=" * 78)

    sd_cov = (build_sd_cache()
                .rename({"subject_id": "participant_id"})
                .to_pandas()[["participant_id", *NEW_SD_COV, "n_days"]])
    log(f"  SD-of-daily covariates available: n = {len(sd_cov):,} participants")

    wp = (pd.read_csv(WITHIN_PERSON_FEATURES)
              .rename(columns={"subject_id": "participant_id"})
              [["participant_id", "SD_daily_mesor",
                "SD_daily_amplitude", "SD_daily_acrophase"]])

    rows: list[dict] = []
    for fname, label in FRAMES:
        log(f"\n=== {label} ===")
        df = pd.read_csv(TABLES_DIR / fname, sep="\t")
        df = df.merge(sd_cov, on="participant_id", how="left")
        df = df.merge(wp,     on="participant_id", how="left")
        log(f"  with new behavioral SDs:    "
            f"n = {df.dropna(subset=NEW_SD_COV).shape[0]:,}")
        log(f"  with within-person rhythm + new SDs: "
            f"n = {df.dropna(subset=NEW_SD_COV + [w[0] for w in WITHIN]).shape[0]:,}")

        for col, plabel in WITHIN:
            sub = df.dropna(subset=NEW_SD_COV + [col]).copy()
            n = len(sub); n_c = int(sub["onset"].sum())
            if n < 50 or n_c < 10:
                log(f"    {plabel:<22s}  <too few>")
                continue
            base = fit_model(sub, [col])
            adj  = fit_model(sub, [col, *NEW_SD_COV])
            # LRT base vs adj (3 added predictors)
            chi2 = 2 * (adj["loglik"] - base["loglik"])
            lrt_p = float(st.chi2.sf(chi2, 3))

            def _or(fit, c):
                b = float(fit["fit"].params[c])
                ci = fit["fit"].conf_int().loc[c].astype(float).tolist()
                return (float(np.exp(b)), float(np.exp(ci[0])),
                        float(np.exp(ci[1])), float(fit["fit"].pvalues[c]))

            ob = _or(base, col); oa = _or(adj, col)
            d_auc = adj["auc"] - base["auc"]
            log(f"    {plabel:<22s}  n = {n:,}, cases = {n_c}")
            log(f"      base: OR = {ob[0]:.2f} [{ob[1]:.2f}, {ob[2]:.2f}], "
                f"p = {ob[3]:.3g},  AUC = {base['auc']:.3f}")
            log(f"      adj : OR = {oa[0]:.2f} [{oa[1]:.2f}, {oa[2]:.2f}], "
                f"p = {oa[3]:.3g},  AUC = {adj['auc']:.3f}  "
                f"(ΔAUC = {d_auc:+.4f}, LRT χ²(3) = {chi2:.2f}, p = {lrt_p:.3g})")

            # ORs for the 3 behavioral SD covariates in the adjusted model
            beh_ors = {c: _or(adj, c) for c in NEW_SD_COV}

            rows.append({
                "outcome": label, "predictor": col, "predictor_label": plabel,
                "n": n, "n_cases": n_c,
                "OR_base":    ob[0], "OR_base_lo": ob[1],
                "OR_base_hi": ob[2], "p_base":     ob[3],
                "auc_base": base["auc"], "aic_base": base["aic"],
                "OR_adj":     oa[0], "OR_adj_lo":  oa[1],
                "OR_adj_hi":  oa[2], "p_adj":      oa[3],
                "auc_adj": adj["auc"], "aic_adj":  adj["aic"],
                "delta_auc": d_auc,
                "lrt_chi2":  chi2, "lrt_df": 3, "lrt_p": lrt_p,
                "or_sd_daily_steps":       beh_ors["sd_daily_steps"][0],
                "p_sd_daily_steps":        beh_ors["sd_daily_steps"][3],
                "or_sd_daily_mets":        beh_ors["sd_daily_mets"][0],
                "p_sd_daily_mets":         beh_ors["sd_daily_mets"][3],
                "or_sd_daily_prop_asleep": beh_ors["sd_daily_prop_asleep"][0],
                "p_sd_daily_prop_asleep":  beh_ors["sd_daily_prop_asleep"][3],
            })

    res = pd.DataFrame(rows)
    res.to_csv(SENS_DIR / "within_person_sd_matched_3signal.tsv",
                sep="\t", index=False)
    log(f"\nWrote {SENS_DIR / 'within_person_sd_matched_3signal.tsv'}")

    # ---------------- Side-by-side comparison vs old 06b (2-covariate) ------
    old_fp = SENS_DIR / "within_person_adjustment.csv"
    if old_fp.exists():
        old = pd.read_csv(old_fp).rename(columns={
            "or_per_sd_base": "OR_base_06b",
            "or_per_sd_adj":  "OR_adj_06b",
            "p_raw_base":     "p_base_06b",
            "p_raw_adj":      "p_adj_06b",
        })[["outcome", "predictor_label",
              "OR_base_06b", "OR_adj_06b",
              "p_base_06b", "p_adj_06b"]]
        merged = res.merge(old, on=["outcome", "predictor_label"], how="left")
        log("\n--- 18c (3-signal) vs 06b (2-covariate) side-by-side ---")
        for _, r in merged.iterrows():
            log(f"  {r['outcome']:<14s} {r['predictor_label']:<22s}  "
                f"06b adj OR = {r['OR_adj_06b']:.2f} (p = {r['p_adj_06b']:.3g})  "
                f"|  18c adj OR = {r['OR_adj']:.2f} (p = {r['p_adj']:.3g}, "
                f"LRT p = {r['lrt_p']:.3g})")

    # ---------------- Markdown summary ----------------
    md = ["# Within-person SD-matched adjustment — three-signal version\n\n"]
    md.append(
        "**Mirrors script 18.** Activity decomposed into separate "
        "SD-of-daily-steps and SD-of-daily-mean-METs; sleep covariate is "
        "SD-of-daily-proportion-of-wear-time-asleep (matching the wear-flag "
        "denominator used by script 18's sleep cosinor).\n\n"
        "Valid-day filter: dairc 4-of-4 quadrant rule (same as script 18). "
        "Per-participant inclusion: ≥7 valid days. Predictors z-scored "
        "within outcome's analytic row set. Models cluster SEs on family. "
        "Base and adjusted fit on identical row sets so any change reflects "
        "the covariates rather than missingness.\n\n"
        "**Framing:** incremental predictive value, not independence. "
        "Within-person variability in daily steps / METs / sleep proportion "
        "is plausibly upstream of within-person variability in HR rhythm, "
        "so the adjusted OR is interpreted as the predictive information "
        "the rhythm SD carries *beyond* the behavioral SDs — not as a "
        "confounder-controlled estimate.\n\n"
    )
    md.append("## Per-outcome results\n\n")
    md.append("| Outcome | Predictor | n / cases | Base OR [95% CI], p | "
                "Adj OR [95% CI], p | ΔAUC | LRT χ²(3), p |\n"
                "|---|---|---|---|---|---|---|\n")
    for _, r in res.iterrows():
        md.append(f"| {r['outcome']} | {r['predictor_label']} | "
                  f"{r['n']:,} / {r['n_cases']} | "
                  f"{r['OR_base']:.2f} [{r['OR_base_lo']:.2f}, "
                  f"{r['OR_base_hi']:.2f}], p = {r['p_base']:.3g} | "
                  f"{r['OR_adj']:.2f} [{r['OR_adj_lo']:.2f}, "
                  f"{r['OR_adj_hi']:.2f}], p = {r['p_adj']:.3g} | "
                  f"{r['delta_auc']:+.4f} | "
                  f"{r['lrt_chi2']:.2f}, p = {r['lrt_p']:.3g} |\n")
    md.append("\n## Behavioral-SD ORs in the adjusted model\n\n")
    md.append("| Outcome | Predictor | SD steps | SD METs | SD prop-asleep |\n"
                "|---|---|---|---|---|\n")
    for _, r in res.iterrows():
        md.append(f"| {r['outcome']} | {r['predictor_label']} | "
                  f"{r['or_sd_daily_steps']:.2f} (p={r['p_sd_daily_steps']:.3g}) | "
                  f"{r['or_sd_daily_mets']:.2f} (p={r['p_sd_daily_mets']:.3g}) | "
                  f"{r['or_sd_daily_prop_asleep']:.2f} "
                  f"(p={r['p_sd_daily_prop_asleep']:.3g}) |\n")
    (SENS_DIR / "within_person_sd_matched_3signal.md").write_text("".join(md))
    log(f"Wrote {SENS_DIR / 'within_person_sd_matched_3signal.md'}")

    (OUTPUTS_DIR / "18c_within_person_sd_matched.log").write_text(
        "\n".join(out_lines))


if __name__ == "__main__":
    main()
