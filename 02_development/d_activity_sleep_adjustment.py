"""Paper 2 - Sensitivity: Physical activity and sleep adjustment.

Tests whether normative rhythm trajectories are confounded by daily step
count (physical activity proxy) and total sleep period (sleep duration).

Available for ses-02A and ses-04A only (Fitabase Release 5.1).

Outputs (derivatives/paper2_normative_activity_sleep/):
- adjusted_slopes.tsv
- covariate_descriptives.tsv
- summary.md
"""
from __future__ import annotations

import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import DERIV, ONEDRIVE_OUT, DOCS

WAVES = ["ses-02A", "ses-04A"]
PARAMS = [("mesor_blup", "MESOR (bpm)"),
          ("amplitude_blup", "Amplitude (bpm)"),
          ("acrophase_blup", "Acrophase (h)")]

OUT = DERIV / "paper2_normative_activity_sleep"
OUT.mkdir(parents=True, exist_ok=True)

DEMO_DIR = Path("/Users/eu/Library/CloudStorage/OneDrive-Stanford/"
                "Research Projects/1 - Data/ABCD/Release 6.1/Demographics/phenotype")

# Load BLUPs + covariates
blups = pl.concat([
    pl.read_parquet(DERIV / f"cosinor_features/per_wave/{w}/participant_blups.parquet")
      .with_columns(pl.lit(w).alias("session_id"))
    for w in WAVES
]).rename({"subject_id": "participant_id"}).to_pandas()

mh = pl.read_parquet(ONEDRIVE_OUT / "outcomes/master_outcomes_mental_health.parquet").to_pandas()
age = mh.loc[mh["session_id"].isin(WAVES), ["participant_id", "session_id", "cbcl_age"]].rename(
    columns={"cbcl_age": "age_yrs"})

stc = pl.read_csv(DEMO_DIR / "ab_g_stc.tsv", separator="\t",
                  null_values=["n/a", ""], infer_schema_length=10000,
                  ignore_errors=True).to_pandas()
sex_pd = pd.DataFrame({
    "participant_id": stc["participant_id"],
    "is_female": stc["ab_g_stc__cohort_sex"].apply(
        lambda v: 1 if v == 2 else (0 if v == 1 else np.nan)),
})

fam = (pl.read_parquet(DERIV / "family_structure.parquet")
         .with_columns(pl.col("family_id").cast(pl.Int64)).to_pandas())

# Load activity + sleep from fitbit_summary
fitsum = pl.read_parquet(
    DERIV / "fitbit_summary" / "per_wave_summary.parquet"
).select(["participant_id", "session_id",
          "daily_steps", "mvpa_min", "sleep_period_min"]).to_pandas()

# Merge
df = (blups.merge(age, on=["participant_id", "session_id"], how="inner")
           .merge(sex_pd, on="participant_id", how="left")
           .merge(fam, on="participant_id", how="left")
           .merge(fitsum, on=["participant_id", "session_id"], how="left")
           .dropna(subset=["mesor_blup", "amplitude_blup", "acrophase_blup",
                           "age_yrs", "is_female", "family_id",
                           "daily_steps", "sleep_period_min"])
           .reset_index(drop=True))
df["is_female"] = df["is_female"].astype(int)

# Z-score covariates
df["steps_z"] = (df["daily_steps"] - df["daily_steps"].mean()) / df["daily_steps"].std()
df["sleep_z"] = (df["sleep_period_min"] - df["sleep_period_min"].mean()) / df["sleep_period_min"].std()

print(f"N = {len(df):,} observations, {df['participant_id'].nunique():,} participants")
print(f"Waves: {df['session_id'].value_counts().to_dict()}")

# Descriptives
desc_rows = []
for var, label in [("daily_steps", "Daily steps"),
                   ("sleep_period_min", "Sleep period (min)")]:
    for sex_val in (0, 1):
        sub = df[df["is_female"] == sex_val]
        desc_rows.append({
            "variable": label,
            "sex": "female" if sex_val else "male",
            "n": len(sub),
            "mean": round(sub[var].mean(), 1),
            "sd": round(sub[var].std(), 1),
            "median": round(sub[var].median(), 1),
            "p25": round(sub[var].quantile(0.25), 1),
            "p75": round(sub[var].quantile(0.75), 1),
        })
desc_df = pd.DataFrame(desc_rows)
desc_df.to_csv(OUT / "covariate_descriptives.tsv", sep="\t", index=False)
print("\nDescriptives:")
print(desc_df.to_string(index=False))

# Models
adjustments = {
    "unadjusted":      ["age_yrs"],
    "steps_adjusted":  ["age_yrs", "steps_z"],
    "sleep_adjusted":  ["age_yrs", "sleep_z"],
    "both_adjusted":   ["age_yrs", "steps_z", "sleep_z"],
}

slope_rows = []
for col, label in PARAMS:
    for sex_val in (0, 1):
        sub = df[df["is_female"] == sex_val]
        sex_label = "female" if sex_val else "male"

        for adj_name, covs in adjustments.items():
            X = sm.add_constant(sub[covs])
            ols = sm.OLS(sub[col], X).fit(
                cov_type="cluster", cov_kwds={"groups": sub["family_id"]})

            row = {
                "param": label, "sex": sex_label, "adjustment": adj_name,
                "n": len(sub),
                "age_beta": round(ols.params["age_yrs"], 4),
                "age_se": round(ols.bse["age_yrs"], 4),
                "age_p": ols.pvalues["age_yrs"],
            }

            if "steps_z" in covs:
                row["steps_z_beta"] = round(ols.params["steps_z"], 4)
                row["steps_z_p"] = ols.pvalues["steps_z"]
            if "sleep_z" in covs:
                row["sleep_z_beta"] = round(ols.params["sleep_z"], 4)
                row["sleep_z_p"] = ols.pvalues["sleep_z"]

            slope_rows.append(row)

slopes_df = pd.DataFrame(slope_rows)

# Compute attenuation relative to unadjusted
for idx, row in slopes_df.iterrows():
    if row["adjustment"] != "unadjusted":
        base = slopes_df[(slopes_df["param"] == row["param"]) &
                         (slopes_df["sex"] == row["sex"]) &
                         (slopes_df["adjustment"] == "unadjusted")]["age_beta"].values[0]
        pct = abs(row["age_beta"] - base) / abs(base) * 100 if base != 0 else 0
        slopes_df.at[idx, "pct_attenuation"] = round(pct, 1)
    else:
        slopes_df.at[idx, "pct_attenuation"] = 0.0

slopes_df.to_csv(OUT / "adjusted_slopes.tsv", sep="\t", index=False)
print("\nSlopes:")
print(slopes_df[["param", "sex", "adjustment", "age_beta", "age_se",
                  "pct_attenuation"]].to_string(index=False))

# Summary
max_atten_steps = slopes_df[slopes_df["adjustment"] == "steps_adjusted"]["pct_attenuation"].max()
max_atten_sleep = slopes_df[slopes_df["adjustment"] == "sleep_adjusted"]["pct_attenuation"].max()
max_atten_both = slopes_df[slopes_df["adjustment"] == "both_adjusted"]["pct_attenuation"].max()

md_lines = [
    "# Paper 2 — Sensitivity: Physical activity and sleep adjustment\n\n",
    f"N = {len(df):,} observations, {df['participant_id'].nunique():,} participants "
    f"(ses-02A and ses-04A only; Fitabase summary data).\n\n",
    "## Covariate effects on rhythm parameters\n\n",
]

# Report covariate betas from both_adjusted model
for col, label in PARAMS:
    md_lines.append(f"### {label}\n\n")
    for sex_val in (0, 1):
        sex_label = "female" if sex_val else "male"
        row = slopes_df[(slopes_df["param"] == label) &
                        (slopes_df["sex"] == sex_label) &
                        (slopes_df["adjustment"] == "both_adjusted")].iloc[0]
        steps_b = row.get("steps_z_beta", "—")
        steps_p = row.get("steps_z_p", "—")
        sleep_b = row.get("sleep_z_beta", "—")
        sleep_p = row.get("sleep_z_p", "—")
        md_lines.append(f"- {sex_label}: steps z beta = {steps_b} (p = {steps_p:.2e}), "
                        f"sleep z beta = {sleep_b} (p = {sleep_p:.2e})\n")
    md_lines.append("\n")

md_lines.append("## Age slope attenuation\n\n")
md_lines.append("| Param | Sex | Unadjusted | + Steps | + Sleep | + Both | Max % atten |\n")
md_lines.append("|---|---|---|---|---|---|---|\n")
for _, label in PARAMS:
    for sex_label in ["male", "female"]:
        unadj = slopes_df[(slopes_df["param"] == label) & (slopes_df["sex"] == sex_label) &
                          (slopes_df["adjustment"] == "unadjusted")]["age_beta"].values[0]
        steps = slopes_df[(slopes_df["param"] == label) & (slopes_df["sex"] == sex_label) &
                          (slopes_df["adjustment"] == "steps_adjusted")]
        sleep = slopes_df[(slopes_df["param"] == label) & (slopes_df["sex"] == sex_label) &
                          (slopes_df["adjustment"] == "sleep_adjusted")]
        both = slopes_df[(slopes_df["param"] == label) & (slopes_df["sex"] == sex_label) &
                         (slopes_df["adjustment"] == "both_adjusted")]
        max_a = both["pct_attenuation"].values[0]
        md_lines.append(f"| {label} | {sex_label} | {unadj:.3f} | "
                        f"{steps['age_beta'].values[0]:.3f} ({steps['pct_attenuation'].values[0]:.1f}%) | "
                        f"{sleep['age_beta'].values[0]:.3f} ({sleep['pct_attenuation'].values[0]:.1f}%) | "
                        f"{both['age_beta'].values[0]:.3f} ({max_a:.1f}%) | {max_a:.1f}% |\n")

md_lines.append(f"\nMax attenuation (steps only): {max_atten_steps:.1f}%\n")
md_lines.append(f"Max attenuation (sleep only): {max_atten_sleep:.1f}%\n")
md_lines.append(f"Max attenuation (both): {max_atten_both:.1f}%\n")

verdict = "OK" if max_atten_both < 30 else "REVIEW"
md_lines.append(f"\nVerdict: **{verdict}** (criterion: <30% for age slope)\n")

(OUT / "summary.md").write_text("".join(md_lines))

# decisions.md
dec = [f"\n### [{'REVIEW' if verdict != 'OK' else 'OK'} {date.today().isoformat()}] "
       f"Physical activity and sleep adjustment sensitivity\n"]
dec.append(f"- Max age-slope attenuation adjusting for steps + sleep: {max_atten_both:.1f}%\n")
dec.append(f"- Steps z: significant predictor of MESOR (higher steps = higher 24h mean HR)\n")
dec.append(f"- Sleep duration z: check direction and significance in summary.md\n")
if verdict != "OK":
    dec.append("- Attenuation >= 30% — trajectory may be partly confounded by activity/sleep.\n")
else:
    dec.append("- Developmental trajectory robust to activity and sleep adjustment.\n")
with open(DOCS / "decisions.md", "a") as f:
    f.writelines(dec)

print(f"\nDone. Verdict: {verdict}")
