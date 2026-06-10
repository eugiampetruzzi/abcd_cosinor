"""Paper 2 - foundational characterization figures.

Builds 4 figures + one supplementary correlation table:
  Fig 1 - developmental trajectories (already rendered by paper2_rhythm_across_age.py;
          we re-render here in the characterization style for consistency)
  Fig 2 - within-person stability (ICC by sex; test-retest scatter W2 -> W4 per param)
  Fig 3 - predictors of within-person change (forest plot per rhythm x sex)
  Fig 4 - cross-feature correlations (between-person + within-person heatmap)

Cross-feature correlations:
  - Between-person: average each pid's BLUPs across waves, then correlate
  - Within-person: deviation of each wave from the participant's average,
    then correlate (captures rhythm-feature co-fluctuation)

Outputs to figures/paper2_characterization/.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from patsy import dmatrix
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import DERIV, ONEDRIVE_OUT  # noqa: E402

FIG_OUT = ONEDRIVE_OUT / "figures" / "paper2_characterization"
FIG_OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT = DERIV / "paper2_rhythm_characterization"

mpl.rcParams.update({
    "font.family": "Arial", "font.sans-serif": ["Arial"],
    "font.size": 12, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.linewidth": 0.9, "axes.edgecolor": "#222222",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "savefig.dpi": 300,
})

MALE = "#4477AA"
FEMALE = "#CC6677"
GREY = "#888888"

PARAMS = [("mesor_blup", "MESOR (bpm)"),
          ("amplitude_blup", "Amplitude (bpm)"),
          ("acrophase_blup", "Acrophase (h)")]
PARAMS_SHORT = ["MESOR", "Amplitude", "Acrophase"]
WAVES = ["ses-02A", "ses-04A", "ses-06A"]


lf = pl.read_parquet(DERIV / "rhythm_trajectory" / "long_form.parquet").to_pandas()
lf["is_female"] = lf["is_female"].astype(float)


# Figure 1: developmental trajectories (cubic spline of age x sex)
print("\nFig 1: developmental trajectories")

df = lf.dropna(subset=["mesor_blup", "amplitude_blup", "acrophase_blup",
                         "age_yrs", "is_female", "participant_id"]).copy()
X_train = dmatrix("cr(age_yrs, df=4) * is_female", df, return_type="dataframe")
di = X_train.design_info
age_grid = np.linspace(df["age_yrs"].min(), df["age_yrs"].max(), 100)

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
for ax, (col, label) in zip(axes, PARAMS):
    md = sm.MixedLM(df[col].to_numpy(), X_train,
                     groups=df["participant_id"].to_numpy()).fit(
                       reml=True, method="lbfgs", maxiter=200)
    beta = md.fe_params.to_numpy()
    cov_fe = md.cov_params().iloc[:X_train.shape[1],
                                   :X_train.shape[1]].to_numpy()
    for sex_val, color, lbl in [(0, MALE, "Male"), (1, FEMALE, "Female")]:
        Xg = dmatrix(di, {"age_yrs": age_grid,
                            "is_female": np.full_like(age_grid, sex_val, dtype=float)},
                       return_type="dataframe").to_numpy()
        mu = Xg @ beta
        se = np.sqrt(np.einsum("ij,jk,ik->i", Xg, cov_fe, Xg))
        ax.fill_between(age_grid, mu - 1.96*se, mu + 1.96*se,
                          color=color, alpha=0.20, linewidth=0)
        ax.plot(age_grid, mu, color=color, linewidth=2.2, label=lbl)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(label)
axes[-1].legend(frameon=False, loc="best")
plt.tight_layout()
plt.savefig(FIG_OUT / "fig1_trajectories.png", bbox_inches="tight")
plt.close()


# Figure 2: within-person stability - ICC bars + W2->W4 test-retest scatter
print("Fig 2: within-person stability")
icc = pd.read_csv(DATA_OUT / "stability_icc.tsv", sep="\t")

fig = plt.figure(figsize=(12.5, 7.0))
gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.4], hspace=0.50, wspace=0.30)
ax_bar = fig.add_subplot(gs[0, :])

x = np.arange(len(PARAMS_SHORT)); off = 0.18
for sex_val, sex_label, color, offset in [(0, "Male", MALE, -off),
                                            (1, "Female", FEMALE, +off)]:
    iccs = []
    for col, _ in PARAMS:
        r = icc[(icc["sex"] == sex_label.lower()) & (icc["param"] == col)]
        iccs.append(float(r["icc"].iloc[0]))
    ax_bar.bar(x + offset, iccs, width=0.32, color=color,
                 edgecolor="white", linewidth=1.0, label=sex_label)
ax_bar.set_xticks(x); ax_bar.set_xticklabels(PARAMS_SHORT)
ax_bar.set_ylabel("ICC(2,1)  (between / total variance)")
ax_bar.set_ylim(0, 0.75)
ax_bar.axhline(0.5, color=GREY, lw=0.6, ls="--", alpha=0.6)
ax_bar.legend(frameon=False, loc="upper right")
ax_bar.text(0.01, 0.97, "More trait-like (higher ICC)  ←      →  More state-like",
             transform=ax_bar.transAxes, fontsize=10, color="#666666",
             ha="left", va="top")

# Test-retest scatter W2 -> W4 per parameter (pooled across sexes for clarity;
# color by sex)
for i, (col, label) in enumerate(PARAMS):
    ax = fig.add_subplot(gs[1, i])
    sub = lf[lf["session_id"].isin(["ses-02A", "ses-04A"])][
        ["participant_id", "session_id", col, "is_female"]].dropna(subset=[col])
    wide = sub.pivot_table(index=["participant_id", "is_female"],
                            columns="session_id",
                            values=col, aggfunc="first").reset_index().dropna(
                              subset=["ses-02A", "ses-04A"])
    for sex_val, color in [(0, MALE), (1, FEMALE)]:
        sw = wide[wide["is_female"] == sex_val]
        ax.scatter(sw["ses-02A"], sw["ses-04A"], s=4, alpha=0.22,
                    color=color, edgecolor="none")
    lo = min(wide["ses-02A"].min(), wide["ses-04A"].min())
    hi = max(wide["ses-02A"].max(), wide["ses-04A"].max())
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(f"{label}  at ses-02A")
    ax.set_ylabel(f"{label}  at ses-04A")
    # annotate r
    r_all = np.corrcoef(wide["ses-02A"], wide["ses-04A"])[0, 1]
    ax.text(0.04, 0.95, f"r = {r_all:.2f}", transform=ax.transAxes,
              fontsize=11, va="top", ha="left", color="#333333")

plt.savefig(FIG_OUT / "fig2_stability.png", bbox_inches="tight")
plt.close()


# Figure 3: forest plot of within-person change predictors
print("Fig 3: predictors of change (forest)")
surv = pd.read_csv(DATA_OUT / "slope_predictor_survey.tsv", sep="\t")

PRED_DISPLAY = {
    "baseline_value": "Baseline value",
    "puberty_timing": "Pubertal timing",
    "puberty_tempo":  "Pubertal tempo",
    "dhea_mean":      "DHEA (mean)",
    "testo_mean":     "Testosterone (mean)",
    "estr_mean":      "Estradiol (mean)",
    "ELA_physical_trauma_z": "ELA (physical trauma)",
}
PRED_ORDER = list(PRED_DISPLAY.keys())

fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.5), sharey=True)
y_pos = np.arange(len(PRED_ORDER))[::-1]
off = 0.18
for ax, (param, plabel) in zip(axes, PARAMS):
    s = surv[surv["param"] == param]
    # Use baseline_value normalized to a comparable scale by dividing by 5
    # (purely visual  keeps it on the same axis as the other betas)
    for sex_label, color, offset in [("male", MALE, -off),
                                       ("female", FEMALE, +off)]:
        sub = s[s["sex"] == sex_label]
        for j, pcol in enumerate(PRED_ORDER):
            r = sub[sub["predictor"] == pcol]
            if r.empty:
                continue
            r = r.iloc[0]
            b = float(r["beta"]); lo = float(r["ci_lo"]); hi = float(r["ci_hi"])
            # Compress baseline value (huge) into smaller scale
            if pcol == "baseline_value":
                continue  # show in a side annotation rather than the panel
            sig = float(r["p"]) < 0.05
            ax.errorbar(b, y_pos[j] + offset, xerr=[[b - lo], [hi - b]],
                          fmt="o", color=color, ecolor=color,
                          markersize=8, capsize=3,
                          markerfacecolor=color if sig else "white",
                          markeredgecolor=color, markeredgewidth=1.4,
                          elinewidth=1.8)
    ax.axvline(0, color=GREY, lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(f"Δ slope per 1-SD predictor\n({plabel} per year)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([PRED_DISPLAY[p] for p in PRED_ORDER])

handles = [
    Line2D([0], [0], marker="o", linestyle="none", color=MALE,
           markersize=9, markerfacecolor=MALE, label="Boys"),
    Line2D([0], [0], marker="o", linestyle="none", color=FEMALE,
           markersize=9, markerfacecolor=FEMALE, label="Girls"),
    Line2D([0], [0], marker="o", linestyle="none", color=GREY,
           markersize=9, markerfacecolor="white", markeredgecolor=GREY,
           markeredgewidth=1.4, label="ns (open)"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
            frameon=False, ncols=3, fontsize=11)
fig.subplots_adjust(bottom=0.20, left=0.16, right=0.97, top=0.92, wspace=0.20)
plt.savefig(FIG_OUT / "fig3_change_predictors.png", bbox_inches="tight")
plt.close()


# Cross-feature correlations + Figure 4 (heatmap)
print("Fig 4: cross-feature correlations")
sub = lf.dropna(subset=["mesor_blup", "amplitude_blup", "acrophase_blup",
                          "is_female", "participant_id"]).copy()

# Between-person: mean each pid's BLUPs across waves
between = (sub.groupby(["participant_id", "is_female"])
              [["mesor_blup", "amplitude_blup", "acrophase_blup"]]
              .mean().reset_index())

# Within-person: deviation from pid mean
sub_dev = sub.copy()
for col, _ in PARAMS:
    pid_means = sub.groupby("participant_id")[col].transform("mean")
    sub_dev[col + "_dev"] = sub[col] - pid_means

rows = []
for sex_val, sex_label in [(0, "male"), (1, "female")]:
    b_sub = between[between["is_female"] == sex_val]
    w_sub = sub_dev[sub_dev["is_female"] == sex_val]
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            ci, cj = PARAMS[i][0], PARAMS[j][0]
            rb = float(np.corrcoef(b_sub[ci], b_sub[cj])[0, 1])
            rw = float(np.corrcoef(w_sub[ci + "_dev"], w_sub[cj + "_dev"])[0, 1])
            rows.append({
                "sex": sex_label,
                "param_i": PARAMS_SHORT[i],
                "param_j": PARAMS_SHORT[j],
                "between_r": rb, "within_r": rw,
                "n_between": int(len(b_sub)),
                "n_within":  int(len(w_sub)),
            })
cross = pd.DataFrame(rows)
cross.to_csv(DATA_OUT / "cross_feature_correlations.tsv",
              sep="\t", index=False)


def matrix(df, kind, sex_label):
    M = np.eye(3)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            r = df[(df["sex"] == sex_label) &
                   (df["param_i"] == PARAMS_SHORT[i]) &
                   (df["param_j"] == PARAMS_SHORT[j])].iloc[0]
            M[i, j] = r[f"{kind}_r"]
    return M


fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.5),
                          gridspec_kw={"hspace": 0.30, "wspace": 0.25})
for col_i, sex_label in enumerate(["male", "female"]):
    for row_i, kind in enumerate(["between", "within"]):
        ax = axes[row_i, col_i]
        M = matrix(cross, kind, sex_label)
        im = ax.imshow(M, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="equal")
        for i in range(3):
            for j in range(3):
                v = M[i, j]
                ax.text(j, i, f"{v:+.2f}" if i != j else "—",
                          ha="center", va="center", fontsize=11,
                          color="white" if abs(v) > 0.4 else "#222222")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(PARAMS_SHORT, rotation=0)
        ax.set_yticklabels(PARAMS_SHORT)
        ax.set_title(f"{kind.capitalize()}-person   ({sex_label})",
                       loc="left", fontsize=11, fontweight="bold",
                       color="#333333", pad=4)
        ax.tick_params(axis="both", which="both", length=0)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6,
                     orientation="vertical", pad=0.04)
cbar.set_label("Pearson r")
plt.savefig(FIG_OUT / "fig4_cross_feature.png", bbox_inches="tight")
plt.close()

print(f"\nWrote 4 figures to {FIG_OUT}/")
