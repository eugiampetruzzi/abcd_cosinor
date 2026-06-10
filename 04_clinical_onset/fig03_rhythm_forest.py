"""Two-panel rhythm forest plot (Hypothesis 2, three-category framing).

Top row  = between-person typical-day cosinor parameters (mesor, amplitude, acrophase).
Bottom row = within-person day-to-day instability (SD of daily mesor, amplitude, acrophase).
Columns are feature-aligned (mesor | amplitude | acrophase). Five outcomes per panel.
Per-1-SD ORs [95% CI] from age+sex-adjusted, family-clustered logistic models vs the
ultra-clean HC pool. Filled marker = FDR-significant (BH across 30 tests); open = ns.
Blue = psychopathology, red = cardiometabolic.
Source: results/trajectory/threecat_between_within.csv.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

BLUE = "#2166AC"
RED = "#B2182B"

SRC = Path("/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/"
           "1 - Data/ABCD/ABCD Actigraphy Resource Paper/fitbit_prediction_superhealthy/"
           "results/trajectory/threecat_between_within.csv")
OUT = Path("/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/"
           "1 - Data/ABCD/ABCD Actigraphy Resource Paper/Longitudinal Paper/combined paper")

ORDER = ["Hypertension", "Obesity", "Externalizing", "Anxiety", "Depression"]
DOMAIN = {"Hypertension": RED, "Obesity": RED,
          "Externalizing": BLUE, "Anxiety": BLUE, "Depression": BLUE}

# (analysis, predictor, column title) for each subplot, row-major
ROWS = [
    ("between", [("Mesor", "Mesor"),
                 ("Amplitude", "Amplitude"),
                 ("Acrophase", "Acrophase")], "Between-person"),
    ("within",  [("SD Mesor", "SD mesor"),
                 ("SD Amplitude", "SD amplitude"),
                 ("SD Acrophase", "SD acrophase")], "Within-person"),
]

d = pd.read_csv(SRC)

fig, axes = plt.subplots(2, 3, figsize=(10, 6.0), sharex=True, sharey=True)
y = np.arange(len(ORDER))

for ri, (analysis, feats, row_label) in enumerate(ROWS):
    da = d[d["analysis"] == analysis]
    for ci, (pred, title) in enumerate(feats):
        ax = axes[ri, ci]
        sub = da[da["predictor"] == pred].set_index("outcome")
        for i, outcome in enumerate(ORDER):
            r = sub.loc[outcome]
            c = DOMAIN[outcome]
            sig = r["p_fdr"] < .05
            ax.plot([r["OR_lo"], r["OR_hi"]], [i, i], color=c, lw=1.4, zorder=1)
            ax.plot(r["OR"], i, marker="o", ms=6, color=c,
                    mfc=c if sig else "white", mec=c, mew=1.4, zorder=2)
        ax.axvline(1.0, color="#999999", lw=0.8, ls="--", zorder=0)
        ax.set_xlim(0.72, 1.78)
        ax.set_xticks([0.8, 1.0, 1.2, 1.4, 1.6])
        ax.set_title(title, fontsize=10, pad=4)
        if ri == 1:
            ax.set_xlabel("OR per 1 SD")
    # row label on the far left of each row
    axes[ri, 0].text(-0.46, 0.5, row_label, transform=axes[ri, 0].transAxes,
                     rotation=90, va="center", ha="center", fontsize=11,
                     fontweight="bold")

for ri in (0, 1):
    axes[ri, 0].set_yticks(y)
    axes[ri, 0].set_yticklabels(ORDER)
axes[0, 0].set_ylim(-0.6, len(ORDER) - 0.4)

legend = [
    Line2D([0], [0], color=BLUE, lw=1.4, marker="o", ms=6, label="Psychopathology"),
    Line2D([0], [0], color=RED, lw=1.4, marker="o", ms=6, label="Cardiometabolic"),
    Line2D([0], [0], color="#444444", lw=0, marker="o", ms=6, mfc="#444444",
           label="FDR p < .05"),
    Line2D([0], [0], color="#444444", lw=0, marker="o", ms=6, mfc="white",
           mec="#444444", mew=1.4, label="ns"),
]
fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False,
           fontsize=9, bbox_to_anchor=(0.5, -0.02), handletextpad=0.4,
           columnspacing=1.6)

fig.tight_layout(rect=(0.03, 0.04, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_rhythm_forest_2panel.{ext}", bbox_inches="tight")
print("Wrote", OUT / "fig_rhythm_forest_2panel.png")
