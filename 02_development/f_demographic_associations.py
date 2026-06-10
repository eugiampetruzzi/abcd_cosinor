"""Paper 2 - Analysis 2: Demographic stratification (race/ethnicity, SES, site).

Tests whether normative rhythm trajectories generalize across demographic
subgroups or whether subgroup-specific curves are needed.

Outputs (derivatives/paper2_normative_demographics/):
- main_effects.tsv
- age_x_subgroup_lrt.tsv
- site_variance_components.tsv
- marginal_predictions_by_subgroup.tsv
- summary.md

Figure (figures/paper2/):
- fig_demographics.png
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
from patsy import dmatrix
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import DERIV, ONEDRIVE_OUT, DOCS

WAVES = ["ses-02A", "ses-04A", "ses-06A"]
PARAMS = [("mesor_blup", "MESOR (bpm)"),
          ("amplitude_blup", "Amplitude (bpm)"),
          ("acrophase_blup", "Acrophase (h)")]

OUT = DERIV / "paper2_normative_demographics"
OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT = ONEDRIVE_OUT / "figures" / "paper2"
FIG_OUT.mkdir(parents=True, exist_ok=True)

DEMO_DIR = Path("/Users/eu/Library/CloudStorage/OneDrive-Stanford/"
                "Research Projects/1 - Data/ABCD/Release 6.1/Demographics/phenotype")
ACTIG_OUT = Path("/Users/eu/Library/CloudStorage/OneDrive-Stanford/"
                 "Research Projects/1 - Data/ABCD/Release 6.1/Actigraphy_Eu_Outputs")

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

# Load demographics
subj_demo = pd.read_csv(ACTIG_OUT / "subject_demographics.tsv", sep="\t")
demo_cols = subj_demo[["participant_id", "ethnrace_label", "income_3lvl_label",
                        "edu_cgs_label", "site_baseline"]].copy()
demo_cols = demo_cols.rename(columns={
    "ethnrace_label": "race_eth",
    "income_3lvl_label": "income_3lvl",
    "edu_cgs_label": "edu_cgs",
    "site_baseline": "site",
})
demo_cols = demo_cols[demo_cols["race_eth"].isin(["White", "Black", "Hispanic", "Asian", "Other"])]

income_map = {"<50k": "Low", "50–100k": "Mid", ">100k": "High"}
demo_cols["ses_tertile"] = demo_cols["income_3lvl"].map(income_map)
demo_cols = demo_cols.dropna(subset=["ses_tertile"])

# Merge
df = (blups.merge(age, on=["participant_id", "session_id"], how="inner")
           .merge(sex_pd, on="participant_id", how="left")
           .merge(fam, on="participant_id", how="left")
           .merge(demo_cols, on="participant_id", how="left")
           .dropna(subset=["mesor_blup", "amplitude_blup", "acrophase_blup",
                           "age_yrs", "is_female", "family_id", "race_eth", "ses_tertile", "site"])
           .reset_index(drop=True))
df["is_female"] = df["is_female"].astype(int)

n_obs = len(df)
n_sub = df["participant_id"].nunique()
print(f"N = {n_obs:,} observations, {n_sub:,} unique participants")
print(f"Race/ethnicity: {df['race_eth'].value_counts().to_dict()}")
print(f"SES: {df['ses_tertile'].value_counts().to_dict()}")
print(f"Sites: {df['site'].nunique()}")

# Model 0: main effects (no interaction with age)
print("\nFitting main-effects models ...")
main_effects_rows = []
lrt_rows = []

for col, label in PARAMS:
    print(f"  {label} ...")
    y = df[col].to_numpy()

    X0 = dmatrix("cr(age_yrs, df=4) * is_female + C(race_eth) + C(ses_tertile)",
                 df, return_type="dataframe")
    md0 = sm.MixedLM(y, X0, groups=df["participant_id"],
                     exog_re=np.ones((len(df), 1)))
    try:
        res0 = md0.fit(reml=True, method="lbfgs", maxiter=200)
    except Exception:
        res0 = md0.fit(reml=True, method="powell", maxiter=300)

    for name in res0.fe_params.index:
        if "race_eth" in name or "ses_tertile" in name:
            main_effects_rows.append({
                "param": label,
                "predictor": name,
                "beta": round(res0.fe_params[name], 4),
                "se": round(res0.bse[name], 4),
                "z": round(res0.tvalues[name], 3),
                "p": res0.pvalues[name],
            })

    # Model 1: interaction with age (does trajectory shape differ by race?)
    X1 = dmatrix("cr(age_yrs, df=4) * is_female * C(race_eth) + C(ses_tertile)",
                 df, return_type="dataframe")
    md1 = sm.MixedLM(y, X1, groups=df["participant_id"],
                     exog_re=np.ones((len(df), 1)))
    try:
        res1 = md1.fit(reml=False, method="lbfgs", maxiter=200)
        res0_ml = md0.fit(reml=False, method="lbfgs", maxiter=200)
        lr_stat = 2 * (res1.llf - res0_ml.llf)
        df_diff = X1.shape[1] - X0.shape[1]
        p_lrt = 1 - stats.chi2.cdf(lr_stat, df_diff)
        lrt_rows.append({
            "param": label,
            "test": "age × race_eth",
            "lr_stat": round(lr_stat, 2),
            "df": df_diff,
            "p": p_lrt,
        })
        print(f"    LRT age×race_eth: chi2={lr_stat:.1f}, df={df_diff}, p={p_lrt:.4f}")
    except Exception as e:
        print(f"    LRT failed: {e}")
        lrt_rows.append({"param": label, "test": "age × race_eth",
                         "lr_stat": np.nan, "df": np.nan, "p": np.nan})

main_df = pd.DataFrame(main_effects_rows)
main_df.to_csv(OUT / "main_effects.tsv", sep="\t", index=False)

lrt_df = pd.DataFrame(lrt_rows)
lrt_df.to_csv(OUT / "age_x_subgroup_lrt.tsv", sep="\t", index=False)

# Site variance components
print("\nSite variance components ...")
site_rows = []
for col, label in PARAMS:
    site_means = df.groupby("site")[col].mean()
    grand_mean = df[col].mean()
    n_per_site = df.groupby("site")[col].count()
    n0 = n_per_site.mean()
    var_between = ((site_means - grand_mean) ** 2).sum() / (len(site_means) - 1)
    var_within = df.groupby("site")[col].var().mean()
    site_icc = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0
    site_rows.append({
        "param": label,
        "n_sites": len(site_means),
        "var_between_sites": round(var_between, 3),
        "var_within_sites": round(var_within, 3),
        "site_icc": round(site_icc, 4),
    })
    print(f"  {label}: site ICC = {site_icc:.4f}")

site_df = pd.DataFrame(site_rows)
site_df.to_csv(OUT / "site_variance_components.tsv", sep="\t", index=False)

# Marginal predictions by race x sex
print("\nComputing marginal predictions ...")
pred_rows = []
age_grid = np.linspace(df["age_yrs"].min(), df["age_yrs"].max(), 60)

for col, label in PARAMS:
    y = df[col].to_numpy()
    X_full = dmatrix("cr(age_yrs, df=4) * is_female + C(race_eth)", df, return_type="dataframe")
    design_info = X_full.design_info

    md = sm.MixedLM(y, X_full, groups=df["participant_id"],
                    exog_re=np.ones((len(df), 1)))
    try:
        res = md.fit(reml=True, method="lbfgs", maxiter=200)
    except Exception:
        res = md.fit(reml=True, method="powell", maxiter=300)

    beta = res.fe_params.to_numpy()
    cov_fe = res.cov_params().iloc[:X_full.shape[1], :X_full.shape[1]].to_numpy()

    for sex_val in (0, 1):
        for race in ["White", "Black", "Hispanic", "Asian", "Other"]:
            Xg = dmatrix(design_info,
                         {"age_yrs": age_grid,
                          "is_female": np.full_like(age_grid, sex_val),
                          "race_eth": np.array([race] * len(age_grid))},
                         return_type="dataframe").to_numpy()
            mu = Xg @ beta
            se = np.sqrt(np.einsum("ij,jk,ik->i", Xg, cov_fe, Xg))
            for a, m, s in zip(age_grid, mu, se):
                pred_rows.append({
                    "param": label, "sex": "female" if sex_val else "male",
                    "race_eth": race, "age_yrs": float(a),
                    "pred": float(m), "lo": float(m - 1.96 * s), "hi": float(m + 1.96 * s),
                })

pred_df = pd.DataFrame(pred_rows)
pred_df.to_csv(OUT / "marginal_predictions_by_subgroup.tsv", sep="\t", index=False)

# Figure: trajectory panels by race x sex
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

RACE_COLORS = {
    "White": "#4477AA", "Black": "#CC6677", "Hispanic": "#DDCC77",
    "Asian": "#117733", "Other": "#888888",
}

fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
for row_i, sex_name in enumerate(["male", "female"]):
    for col_i, (_, label) in enumerate(PARAMS):
        ax = axes[row_i, col_i]
        for race, color in RACE_COLORS.items():
            sub = pred_df[(pred_df["param"] == label) &
                          (pred_df["sex"] == sex_name) &
                          (pred_df["race_eth"] == race)].sort_values("age_yrs")
            ax.plot(sub["age_yrs"], sub["pred"], color=color, linewidth=1.8, label=race)
            ax.fill_between(sub["age_yrs"], sub["lo"], sub["hi"],
                            color=color, alpha=0.1, linewidth=0)
        if row_i == 1:
            ax.set_xlabel("Age (years)")
        if col_i == 0:
            ax.set_ylabel(f"{'Male' if row_i == 0 else 'Female'}\n{label}")
        else:
            ax.set_ylabel(label)
        if row_i == 0:
            ax.set_title(label.split("(")[0].strip())

axes[0, 2].legend(frameon=False, loc="best", fontsize=8)
plt.tight_layout()
fig_path = FIG_OUT / "fig_demographics.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nFigure: {fig_path}")

# Summary
md_lines = [
    "# Paper 2 — Analysis 2: Demographic stratification\n\n",
    f"N = {n_obs:,} observations, {n_sub:,} unique participants.\n\n",
    f"Race/ethnicity: {df['race_eth'].value_counts().to_dict()}\n\n",
    "## Main effects (race/ethnicity, SES)\n\n",
    "| Param | Predictor | beta | SE | z | p |\n|---|---|---|---|---|---|\n",
]
for _, r in main_df.iterrows():
    md_lines.append(f"| {r['param']} | {r['predictor']} | {r['beta']:.3f} | "
                    f"{r['se']:.3f} | {r['z']:.2f} | {r['p']:.3e} |\n")

md_lines.append("\n## LRT: age × race/ethnicity interaction\n\n")
md_lines.append("| Param | LR stat | df | p |\n|---|---|---|---|\n")
for _, r in lrt_df.iterrows():
    md_lines.append(f"| {r['param']} | {r['lr_stat']:.1f} | {int(r['df'])} | {r['p']:.4f} |\n")

md_lines.append("\n## Site ICC\n\n")
md_lines.append("| Param | Site ICC |\n|---|---|\n")
for _, r in site_df.iterrows():
    md_lines.append(f"| {r['param']} | {r['site_icc']:.4f} |\n")

any_sig_lrt = any(lrt_df["p"] < 0.05)
md_lines.append(f"\n## Verdict\n")
if any_sig_lrt:
    md_lines.append("- Age × race/ethnicity LRT significant for at least one parameter — "
                    "investigate whether subgroup-specific norms are needed.\n")
else:
    md_lines.append("- No significant age × race/ethnicity interaction — "
                    "single normative curves adequate across subgroups.\n")

(OUT / "summary.md").write_text("".join(md_lines))

# decisions.md
dec = [f"\n### [REVIEW {date.today().isoformat()}] Demographic stratification (Analysis 2)\n"]
for _, r in lrt_df.iterrows():
    dec.append(f"- {r['param']} age×race_eth LRT: p = {r['p']:.4f}\n")
if any_sig_lrt:
    dec.append("- Significant interactions detected. Decide: subgroup-specific norms vs. note in supplement.\n")
else:
    dec.append("- No significant interactions. Pooled norms adequate.\n")
with open(DOCS / "decisions.md", "a") as f:
    f.writelines(dec)

print("Done.")
