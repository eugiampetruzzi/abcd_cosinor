"""Paper 2 - Analysis 1: GAMLSS normative centile curves (Python wrapper).

Calls paper2_normative_centiles.R via subprocess, then reads the centile
outputs and generates the manuscript figure (2 rows x 3 cols: sex x param,
P3-P97 fan with scatter overlay).

Outputs:
- derivatives/paper2_normative_centiles/   (written by R)
- figures/paper2/fig_normative_centiles.png (written here)
- decisions.md entry for distribution family choice
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import DERIV, ONEDRIVE_OUT, DOCS

OUT = DERIV / "paper2_normative_centiles"
FIG_OUT = ONEDRIVE_OUT / "figures" / "paper2"
FIG_OUT.mkdir(parents=True, exist_ok=True)

# Run R
r_script = Path(__file__).with_suffix(".R")
print(f"Running {r_script.name} ...")
result = subprocess.run(
    ["Rscript", str(r_script), str(DERIV)],
    capture_output=True, text=True, timeout=1800
)
print(result.stdout)
if result.returncode != 0:
    print("R STDERR:", result.stderr)
    sys.exit(1)

# Load centiles and raw data for figure
centiles = pd.read_csv(OUT / "centiles_all.tsv")
dist_sel = pd.read_csv(OUT / "distribution_selection.tsv")

WAVES = ["ses-02A", "ses-04A", "ses-06A"]
blups = pl.concat([
    pl.read_parquet(DERIV / f"cosinor_features/per_wave/{w}/participant_blups.parquet")
      .with_columns(pl.lit(w).alias("session_id"))
    for w in WAVES
]).rename({"subject_id": "participant_id"}).to_pandas()

mh = pl.read_parquet(ONEDRIVE_OUT / "outcomes/master_outcomes_mental_health.parquet").to_pandas()
age = mh.loc[mh["session_id"].isin(WAVES), ["participant_id", "session_id", "cbcl_age"]].rename(
    columns={"cbcl_age": "age_yrs"})

DEMO_DIR = Path("/Users/eu/Library/CloudStorage/OneDrive-Stanford/"
                "Research Projects/1 - Data/ABCD/Release 6.1/Demographics/phenotype")
stc = pl.read_csv(DEMO_DIR / "ab_g_stc.tsv", separator="\t",
                  null_values=["n/a", ""], infer_schema_length=10000,
                  ignore_errors=True).to_pandas()
sex_map = dict(zip(stc["participant_id"],
                   stc["ab_g_stc__cohort_sex"].map({1: "male", 2: "female"})))

df = blups.merge(age, on=["participant_id", "session_id"], how="inner")
df["sex"] = df["participant_id"].map(sex_map)
df = df.dropna(subset=["mesor_blup", "age_yrs", "sex"])

# Figure: 2 rows (sex) x 3 cols (param)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

params = [("MESOR", "mesor_blup", "MESOR (bpm)"),
          ("Amplitude", "amplitude_blup", "Amplitude (bpm)"),
          ("Acrophase", "acrophase_blup", "Acrophase")]
sexes = [("male", "Male"), ("female", "Female")]

SEX_BASE = {"male": "#4477AA", "female": "#CC6677"}
BAND_ALPHAS = {
    (3, 97):  0.12,
    (10, 90): 0.20,
    (25, 75): 0.30,
}
BAND_LABELS = {
    (3, 97):  "3rd–97th",
    (10, 90): "10th–90th",
    (25, 75): "25th–75th",
}

def acro_fix(v):
    return np.where(v < 6, v + 24, v)

def hours_to_clocklabel(h):
    h = h % 24
    hr = int(h)
    suffix = "AM" if hr < 12 else "PM"
    hr12 = hr % 12
    if hr12 == 0:
        hr12 = 12
    return f"{hr12} {suffix}"

fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)

for row_i, (sx, sx_label) in enumerate(sexes):
    color = SEX_BASE[sx]
    for col_i, (par_label, par_col, ylabel) in enumerate(params):
        ax = axes[row_i, col_i]
        sub_c = centiles[(centiles["param"] == par_label) & (centiles["sex"] == sx)]
        is_acro = par_col == "acrophase_blup"

        for (lo_c, hi_c), alpha in BAND_ALPHAS.items():
            lo_vals = sub_c[sub_c["centile"] == lo_c].sort_values("age_yrs")
            hi_vals = sub_c[sub_c["centile"] == hi_c].sort_values("age_yrs")
            lo_v = acro_fix(lo_vals["value"].values) if is_acro else lo_vals["value"].values
            hi_v = acro_fix(hi_vals["value"].values) if is_acro else hi_vals["value"].values
            ax.fill_between(lo_vals["age_yrs"].values, lo_v, hi_v,
                            color=color, alpha=alpha, linewidth=0,
                            label=BAND_LABELS[(lo_c, hi_c)] if row_i == 0 and col_i == 0 else None)

        for cc in [3, 10, 25, 75, 90, 97]:
            pc = sub_c[sub_c["centile"] == cc].sort_values("age_yrs")
            pc_v = acro_fix(pc["value"].values) if is_acro else pc["value"].values
            lw = 0.8
            ax.plot(pc["age_yrs"], pc_v, color=color, linewidth=lw, alpha=0.5)

        p50 = sub_c[sub_c["centile"] == 50].sort_values("age_yrs")
        p50_v = acro_fix(p50["value"].values) if is_acro else p50["value"].values
        ax.plot(p50["age_yrs"], p50_v, color=color, linewidth=2.5,
                label="50th" if row_i == 0 and col_i == 0 else None)

        # Right-side percentile labels at age 18
        for cc, offset in [(97, 0), (90, 0), (75, 0), (50, 0), (25, 0), (10, 0), (3, 0)]:
            pc = sub_c[sub_c["centile"] == cc].sort_values("age_yrs")
            if len(pc) == 0:
                continue
            last_age = pc["age_yrs"].values[-1]
            last_val = pc["value"].values[-1]
            if is_acro:
                last_val = last_val + 24 if last_val < 6 else last_val
            ordinal = {3: "3rd", 10: "10th", 25: "25th", 50: "50th",
                       75: "75th", 90: "90th", 97: "97th"}[cc]
            fontsize = 7.5 if cc == 50 else 6.5
            fontweight = "bold" if cc == 50 else "normal"
            ax.text(last_age + 0.15, last_val, ordinal, fontsize=fontsize,
                    fontweight=fontweight, color=color, alpha=0.8,
                    va="center", ha="left")

        if is_acro:
            ymin, ymax = ax.get_ylim()
            tick_hours = [h for h in [12, 13, 14, 15, 16, 17, 18, 19, 20] if ymin <= h <= ymax]
            ax.set_yticks(tick_hours)
            ax.set_yticklabels([hours_to_clocklabel(h) for h in tick_hours])

        if row_i == 1:
            ax.set_xlabel("Age (years)")
        if col_i == 0:
            ax.set_ylabel(f"{sx_label}\n{ylabel}")
        else:
            ax.set_ylabel(ylabel)

        if row_i == 0:
            ax.set_title(par_label)

        ax.margins(x=0.08)

plt.tight_layout()
fig_path = FIG_OUT / "fig_normative_centiles.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved: {fig_path}")

# Validate P50 against rhythm_across_age mean
rhythm_preds = pd.read_csv(DERIV / "paper2_rhythm_across_age" / "marginal_predictions.tsv",
                           sep="\t")
lookup = pd.read_csv(ONEDRIVE_OUT / "tables" / "T_normative_lookup.tsv")

print("\nP50 vs mean trajectory at age 12 and 15:")
for par_label, par_col, _ in params:
    for sx_val, sx_name in [(0, "male"), (1, "female")]:
        for check_age in [12.0, 15.0]:
            rp = rhythm_preds[(rhythm_preds["param"] == par_col) &
                              (rhythm_preds["is_female"] == sx_val)]
            closest = rp.iloc[(rp["age_yrs"] - check_age).abs().argsort()[:1]]
            mean_traj = closest["pred"].values[0]

            p50_row = lookup[(lookup["param"] == par_label) &
                             (lookup["sex"] == sx_name) &
                             (lookup["age_yrs"] == check_age) &
                             (lookup["centile"] == 50)]
            if len(p50_row) > 0:
                p50_val = p50_row["value"].values[0]
                diff = abs(p50_val - mean_traj)
                status = "OK" if diff < 1.0 else "REVIEW"
                print(f"  {par_label} {sx_name} age {check_age}: P50={p50_val:.1f}, "
                      f"mean={mean_traj:.1f}, diff={diff:.2f} [{status}]")

# Log to decisions.md
best_families = (dist_sel.sort_values("gaic")
                 .groupby(["param", "sex"]).first().reset_index())
dec_lines = [
    f"\n### [REVIEW {date.today().isoformat()}] GAMLSS distribution family selection (Analysis 1)\n",
]
for _, r in best_families.iterrows():
    dec_lines.append(f"- {r['param']} / {r['sex']}: {r['family']} (GAIC = {r['gaic']:.1f})\n")
dec_lines.append("- Selected by GAIC with k=log(N); candidates: BCT, BCCG, NO.\n")
dec_lines.append("- Worm plots should be checked visually before finalizing.\n")

decisions_path = DOCS / "decisions.md"
with open(decisions_path, "a") as f:
    f.writelines(dec_lines)
print(f"\nLogged distribution selection to {decisions_path}")
