"""Paper 2 - foundational characterization of cardiac rhythm across development.

Three sections:
  1. Within-person stability of each cosinor BLUP across waves
       - ICC(2,1) per rhythm x sex
       - Pairwise cross-wave Pearson r per rhythm x wave-pair x sex
       - Variance decomposition: between vs within
  2. Per-participant change scores
       - For participants with >=2 waves, OLS slope of rhythm ~ age
       - Distribution by sex
  3. Predictor survey of slope (sex-stratified, univariable)
       - baseline_age, baseline_rhythm
       - pubertal timing, pubertal tempo (youth PDS)
       - mean hormone z (DHEA, testo, estradiol-girls)
       - ELA physical trauma z

Outputs (derivatives/paper2_rhythm_characterization/):
- stability_icc.tsv
- stability_cross_wave_r.tsv
- per_participant_slopes.parquet
- slope_predictor_survey.tsv
- summary.md
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
from scipy import stats as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import DERIV  # noqa: E402

OUT = DERIV / "paper2_rhythm_characterization"
OUT.mkdir(parents=True, exist_ok=True)

ELA_FP = Path("/Users/eu/Library/CloudStorage/OneDrive-Stanford/"
               "Research Projects/1 - Data/ABCD/Release 5.1/"
               "early life adversity/ELA_final.csv")

PARAMS = [("mesor_blup", "MESOR (bpm)"),
          ("amplitude_blup", "Amplitude (bpm)"),
          ("acrophase_blup", "Acrophase (h)")]
WAVES = ["ses-02A", "ses-04A", "ses-06A"]


lf = pl.read_parquet(DERIV / "rhythm_trajectory" / "long_form.parquet").to_pandas()
lf["is_female"] = lf["is_female"].astype(float)
print(f"Long form: {len(lf):,} obs, {lf['participant_id'].nunique():,} pids")


# 1. Within-person stability
print("\n=== 1. Within-person stability ===")
icc_rows = []
xwave_rows = []
for y, ylabel in PARAMS:
    for sex_val, sex_label in [(0, "male"), (1, "female")]:
        sub = lf[lf["is_female"] == sex_val][
            ["participant_id", "session_id", y]].dropna(subset=[y])
        if sub["participant_id"].nunique() < 50:
            continue

        # ICC via variance components
        try:
            md = smf.mixedlm(f"{y} ~ 1", data=sub,
                              groups="participant_id").fit(
                                  method="lbfgs", reml=True, maxiter=200)
            sigma_b2 = float(md.cov_re.iloc[0, 0])
            sigma_w2 = float(md.scale)
            icc = sigma_b2 / (sigma_b2 + sigma_w2)
            pct_between = 100 * sigma_b2 / (sigma_b2 + sigma_w2)
            icc_rows.append({
                "param": y, "param_label": ylabel, "sex": sex_label,
                "n_obs": int(len(sub)),
                "n_pid": int(sub["participant_id"].nunique()),
                "variance_between": sigma_b2,
                "variance_within":  sigma_w2,
                "icc": icc, "pct_between": pct_between,
            })
        except Exception as e:
            print(f"ICC failed for {y} {sex_label}: {e}")

        # Pairwise cross-wave Pearson r
        wide = sub.pivot_table(index="participant_id", columns="session_id",
                                values=y, aggfunc="first").reset_index()
        for w1, w2 in [("ses-02A", "ses-04A"), ("ses-04A", "ses-06A"),
                        ("ses-02A", "ses-06A")]:
            if w1 not in wide.columns or w2 not in wide.columns:
                continue
            both = wide[[w1, w2]].dropna()
            if len(both) < 30:
                continue
            r = st.pearsonr(both[w1], both[w2])
            xwave_rows.append({
                "param": y, "param_label": ylabel, "sex": sex_label,
                "wave_pair": f"{w1}->{w2}", "n_pairs": int(len(both)),
                "pearson_r": float(r[0]), "p": float(r[1]),
            })

pd.DataFrame(icc_rows).to_csv(OUT / "stability_icc.tsv", sep="\t", index=False)
pd.DataFrame(xwave_rows).to_csv(OUT / "stability_cross_wave_r.tsv",
                                  sep="\t", index=False)


# 2. Per-participant change scores (multi-wave subjects)
print("\n=== 2. Per-participant slopes ===")
slope_rows = []
for sex_val, sex_label in [(0, "male"), (1, "female")]:
    sub = lf[lf["is_female"] == sex_val]
    for y, ylabel in PARAMS:
        for pid, g in sub.groupby("participant_id"):
            g = g.dropna(subset=[y, "age_yrs"])
            if len(g) < 2:
                continue
            X = sm = np.column_stack([np.ones(len(g)), g["age_yrs"].values])
            try:
                beta, _, _, _ = np.linalg.lstsq(X, g[y].values, rcond=None)
                slope_rows.append({
                    "participant_id": pid, "sex": sex_label,
                    "is_female": sex_val,
                    "param": y, "param_label": ylabel,
                    "n_waves": int(len(g)),
                    "baseline_age": float(g["age_yrs"].min()),
                    "baseline_value": float(g.loc[g["age_yrs"].idxmin(), y]),
                    "mean_value": float(g[y].mean()),
                    "slope_per_year": float(beta[1]),
                })
            except Exception:
                pass

slopes = pd.DataFrame(slope_rows)
slopes.to_parquet(OUT / "per_participant_slopes.parquet")
print(f"  N slopes: {len(slopes):,} (per param * pid * sex)")


# Quick descriptive: slope distribution by sex and rhythm
desc_rows = []
for (sex_label, param), g in slopes.groupby(["sex", "param"]):
    desc_rows.append({
        "sex": sex_label, "param": param,
        "param_label": dict(PARAMS)[param],
        "n_pid": int(g["participant_id"].nunique()),
        "slope_mean": float(g["slope_per_year"].mean()),
        "slope_sd": float(g["slope_per_year"].std()),
        "slope_p25": float(g["slope_per_year"].quantile(0.25)),
        "slope_median": float(g["slope_per_year"].median()),
        "slope_p75": float(g["slope_per_year"].quantile(0.75)),
    })
pd.DataFrame(desc_rows).to_csv(OUT / "slope_descriptives.tsv",
                                 sep="\t", index=False)


# 3. Predictor survey: what predicts slope?
print("\n=== 3. Predictor survey ===")

# Pull predictors
timing = pl.read_parquet(DERIV / "paper2_timing_tempo" /
                           "timing_tempo_youth.parquet").to_pandas()
timing = timing[["participant_id", "timing", "tempo"]].rename(
    columns={"timing": "puberty_timing", "tempo": "puberty_tempo"})

horm = (lf.groupby("participant_id")
          .agg(dhea_mean=("phs_dhea_z", "mean"),
                testo_mean=("phs_testosterone_z", "mean"),
                estr_mean=("phs_estradiol_z", "mean"))
          .reset_index())

ela = pd.read_csv(ELA_FP)
ela["participant_id"] = ela["src_subject_id"].str.replace("NDAR_INV", "sub-", regex=False)
ela = ela[["participant_id", "ELA_physical_trauma"]].drop_duplicates("participant_id")
ela["ELA_physical_trauma_z"] = (ela["ELA_physical_trauma"]
                                  - ela["ELA_physical_trauma"].mean()) / \
                                  ela["ELA_physical_trauma"].std()
ela = ela[["participant_id", "ELA_physical_trauma_z"]]

# Build predictor frame
preds = slopes.merge(timing, on="participant_id", how="left") \
              .merge(horm,   on="participant_id", how="left") \
              .merge(ela,    on="participant_id", how="left")

PRED_LIST = [
    "baseline_age", "baseline_value",
    "puberty_timing", "puberty_tempo",
    "dhea_mean", "testo_mean", "estr_mean",
    "ELA_physical_trauma_z",
]

# Z-score predictors within sex for interpretability
for sex_val in (0, 1):
    mask = preds["is_female"] == sex_val
    for c in PRED_LIST:
        m = preds.loc[mask, c].mean()
        s = preds.loc[mask, c].std()
        if s and not np.isnan(s) and s > 0:
            preds.loc[mask, f"{c}_z"] = (preds.loc[mask, c] - m) / s
        else:
            preds.loc[mask, f"{c}_z"] = np.nan


surv_rows = []
for param, plabel in PARAMS:
    for sex_val, sex_label in [(0, "male"), (1, "female")]:
        sub = preds[(preds["param"] == param) & (preds["is_female"] == sex_val)]
        for pcol in PRED_LIST:
            if pcol == "estr_mean" and sex_val == 0:
                continue
            x_col = f"{pcol}_z"
            d = sub.dropna(subset=["slope_per_year", x_col])
            if len(d) < 50:
                continue
            X = np.column_stack([np.ones(len(d)), d[x_col].values])
            y = d["slope_per_year"].values
            # OLS with HC0 SE
            res = smf.ols("slope_per_year ~ " + x_col, data=d).fit()
            b = float(res.params[x_col]); se = float(res.bse[x_col])
            p = float(res.pvalues[x_col])
            surv_rows.append({
                "sex": sex_label, "param": param, "param_label": plabel,
                "predictor": pcol, "n": int(len(d)),
                "beta": b, "se": se,
                "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se,
                "p": p,
            })

surv = pd.DataFrame(surv_rows)
surv.to_csv(OUT / "slope_predictor_survey.tsv", sep="\t", index=False)


# Markdown summary
md = ["# Paper 2 - foundational characterization of cardiac rhythm across development\n\n",
      "## 1. Within-person stability\n\n",
      "ICC(2,1) = between-person variance / total variance, from a random-intercept "
      "mixed model (`rhythm ~ 1 + (1|pid)`). Higher ICC = more trait-like.\n\n",
      "| Param | Sex | N obs / pid | between | within | ICC | % between |\n",
      "|---|---|---|---|---|---|---|\n"]
for r in icc_rows:
    md.append(f"| {r['param_label']} | {r['sex']} | "
              f"{r['n_obs']:,} / {r['n_pid']:,} | "
              f"{r['variance_between']:.2f} | {r['variance_within']:.2f} | "
              f"{r['icc']:.3f} | {r['pct_between']:.1f}% |\n")

md.append("\n### Cross-wave Pearson r (pairwise)\n\n"
          "| Param | Sex | Wave pair | N pairs | r |\n|---|---|---|---|---|\n")
for r in xwave_rows:
    md.append(f"| {r['param_label']} | {r['sex']} | {r['wave_pair']} | "
              f"{r['n_pairs']:,} | {r['pearson_r']:+.3f} |\n")

md.append("\n## 2. Per-participant slopes (multi-wave subjects)\n\n"
          "Per-pid OLS slope of rhythm ~ age, fit on >=2-wave subjects.\n\n"
          "| Param | Sex | N pid | Slope mean | Slope SD | Median | "
          "p25 -> p75 |\n|---|---|---|---|---|---|---|\n")
for r in desc_rows:
    md.append(f"| {r['param_label']} | {r['sex']} | {r['n_pid']:,} | "
              f"{r['slope_mean']:+.3f} | {r['slope_sd']:.3f} | "
              f"{r['slope_median']:+.3f} | "
              f"{r['slope_p25']:+.3f} -> {r['slope_p75']:+.3f} |\n")

md.append("\n## 3. Predictors of within-person change (slope as outcome)\n\n"
          "Univariable OLS, sex-stratified. Predictors z-scored within sex; "
          "beta is the slope-change per 1-SD higher predictor.\n\n"
          "| Sex | Param | Predictor | N | beta | 95% CI | p |\n"
          "|---|---|---|---|---|---|---|\n")
for r in surv_rows:
    md.append(f"| {r['sex']} | {r['param_label']} | {r['predictor']} | "
              f"{r['n']:,} | {r['beta']:+.4f} | "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p']:.2g} |\n")

(OUT / "summary.md").write_text("".join(md))
print(f"\nWrote {OUT}/")
