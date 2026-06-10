"""Paper 2 - descriptive section: cosinor rhythm trajectory across age x sex.

Pools BLUPs across ses-02A / ses-04A / ses-06A and fits:
    param ~ cr(age_yrs, df=4) * is_female,  random intercept per participant.

Linear slope per sex reported separately via OLS with cluster-robust SE on
family_id (matches age_sex_audit.py convention).

Outputs (derivatives/paper2_rhythm_across_age/):
- marginal_predictions.tsv   age grid x sex with predicted mean and 95% CI
- linear_slopes_by_sex.tsv   linear beta/yr per sex per param
- summary.md                 readable

Figure (figures/paper2/):
- fig_rhythm_across_age.png  three panels (MESOR / Amplitude / Acrophase)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from patsy import dmatrix
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import DERIV, ONEDRIVE_OUT  # noqa: E402

WAVES = ["ses-02A", "ses-04A", "ses-06A"]
PARAMS = [("mesor_blup", "MESOR (bpm)"),
          ("amplitude_blup", "Amplitude (bpm)"),
          ("acrophase_blup", "Acrophase (h)")]
DEMO_DIR = Path("/Users/eu/Library/CloudStorage/OneDrive-Stanford/"
                "Research Projects/1 - Data/ABCD/Release 6.1/Demographics/phenotype")

OUT = DERIV / "paper2_rhythm_across_age"
OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT = ONEDRIVE_OUT / "figures" / "paper2"
FIG_OUT.mkdir(parents=True, exist_ok=True)


blups = pl.concat([
    pl.read_parquet(DERIV / f"cosinor_features/per_wave/{w}/participant_blups.parquet")
      .with_columns(pl.lit(w).alias("session_id"))
    for w in WAVES
]).rename({"subject_id": "participant_id"}).to_pandas()

mh = pl.read_parquet(ONEDRIVE_OUT / "outcomes/master_outcomes_mental_health.parquet").to_pandas()
age = (mh.loc[mh["session_id"].isin(WAVES), ["participant_id", "session_id", "cbcl_age"]]
         .rename(columns={"cbcl_age": "age_yrs"}))

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

df = (blups.merge(age, on=["participant_id", "session_id"], how="inner")
            .merge(sex_pd, on="participant_id", how="left")
            .merge(fam, on="participant_id", how="left")
            .dropna(subset=["mesor_blup", "amplitude_blup", "acrophase_blup",
                            "age_yrs", "is_female", "family_id"])
            .reset_index(drop=True))
df["is_female"] = df["is_female"].astype(int)

n_obs = len(df)
n_sub = df["participant_id"].nunique()
n_wave = df.groupby("session_id").size().to_dict()
print(f"N observations = {n_obs:,}  unique participants = {n_sub:,}")
print(f"Per wave: {n_wave}")


X_train = dmatrix("cr(age_yrs, df=4) * is_female", df, return_type="dataframe")
design_info = X_train.design_info

age_grid = np.linspace(df["age_yrs"].min(), df["age_yrs"].max(), 100)

slope_rows = []
pred_rows = []

for col, label in PARAMS:
    y = df[col].to_numpy()
    md = sm.MixedLM(y, X_train, groups=df["participant_id"].to_numpy())
    res = md.fit(reml=True, method="lbfgs", maxiter=200)

    cov_fe = res.cov_params().iloc[:X_train.shape[1], :X_train.shape[1]].to_numpy()
    beta = res.fe_params.to_numpy()

    for sex_val in (0, 1):
        Xg = dmatrix(design_info,
                     {"age_yrs": age_grid,
                      "is_female": np.full_like(age_grid, sex_val, dtype=float)},
                     return_type="dataframe").to_numpy()
        mu = Xg @ beta
        se = np.sqrt(np.einsum("ij,jk,ik->i", Xg, cov_fe, Xg))
        for a, m, s in zip(age_grid, mu, se):
            pred_rows.append({
                "param": col, "label": label, "is_female": sex_val,
                "age_yrs": float(a), "pred": float(m),
                "lo": float(m - 1.96 * s), "hi": float(m + 1.96 * s),
            })

    for sex_val in (0, 1):
        sub = df[df["is_female"] == sex_val]
        Xs = sm.add_constant(sub[["age_yrs"]])
        fs = sm.OLS(sub[col].to_numpy(), Xs).fit(
            cov_type="cluster", cov_kwds={"groups": sub["family_id"]})
        slope_rows.append({
            "param": col, "label": label,
            "sex": "female" if sex_val == 1 else "male",
            "n_obs": int(len(sub)),
            "n_participants": int(sub["participant_id"].nunique()),
            "linear_beta_per_yr": float(fs.params["age_yrs"]),
            "se": float(fs.bse["age_yrs"]),
            "ci_lo": float(fs.conf_int().loc["age_yrs", 0]),
            "ci_hi": float(fs.conf_int().loc["age_yrs", 1]),
            "p": float(fs.pvalues["age_yrs"]),
        })

slopes_df = pd.DataFrame(slope_rows)
preds_df = pd.DataFrame(pred_rows)
slopes_df.to_csv(OUT / "linear_slopes_by_sex.tsv", sep="\t", index=False)
preds_df.to_csv(OUT / "marginal_predictions.tsv", sep="\t", index=False)


plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

MALE = "#4477AA"
FEMALE = "#CC6677"

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
for ax, (col, label) in zip(axes, PARAMS):
    sub = preds_df[preds_df["param"] == col]
    for sex_val, color, lbl in [(0, MALE, "Male"), (1, FEMALE, "Female")]:
        s = sub[sub["is_female"] == sex_val].sort_values("age_yrs")
        ax.fill_between(s["age_yrs"], s["lo"], s["hi"],
                        color=color, alpha=0.20, linewidth=0)
        ax.plot(s["age_yrs"], s["pred"], color=color, linewidth=2.0, label=lbl)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(label)
axes[-1].legend(frameon=False, loc="best")
plt.tight_layout()
plt.savefig(FIG_OUT / "fig_rhythm_across_age.png", dpi=300, bbox_inches="tight")
plt.close()


md = [
    "# Paper 2 - Rhythm across age (descriptive)\n\n",
    f"N observations = {n_obs:,} across {n_sub:,} unique participants.\n\n",
    f"Per wave: {', '.join(f'{k}={v:,}' for k, v in sorted(n_wave.items()))}.\n\n",
    "Model: cosinor BLUP ~ cr(age_yrs, df=4) * is_female; random intercept per participant.\n",
    "Linear slope per year reported per-sex from OLS with cluster-robust SE on family_id.\n\n",
    "## Linear slope per year by sex\n\n",
    "| Param | Sex | beta/yr | SE | 95% CI | p |\n",
    "|---|---|---|---|---|---|\n",
]
for r in slope_rows:
    md.append(
        f"| {r['label']} | {r['sex']} | {r['linear_beta_per_yr']:+.3f} | "
        f"{r['se']:.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['p']:.3g} |\n"
    )
(OUT / "summary.md").write_text("".join(md))
print(f"\nWrote {OUT} and {FIG_OUT}/fig_rhythm_across_age.png")
