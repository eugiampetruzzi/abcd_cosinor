"""Paper 2 - methods-anchor: within-person hormone coupling to cosinor rhythm.

Reads the existing between/within decomposition coefficients
(model `3_between_within` in derivatives/rhythm_trajectory/coefficients.tsv,
formula: rhythm ~ age_c + hormone_mean + hormone_dev + is_female,
random intercept per participant)
and assembles a tidy table + forest figure isolating the within-person (state)
versus between-person (trait) hormone-rhythm associations.

Outputs (derivatives/paper2_within_person_coupling/):
- between_within_table.tsv
- summary.md

Figure (figures/paper2/):
- fig_within_person_coupling.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import DERIV, ONEDRIVE_OUT  # noqa: E402

OUT = DERIV / "paper2_within_person_coupling"
OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT = ONEDRIVE_OUT / "figures" / "paper2"
FIG_OUT.mkdir(parents=True, exist_ok=True)

COEF_PATH = DERIV / "rhythm_trajectory" / "coefficients.tsv"
df = pd.read_csv(COEF_PATH, sep="\t")
bw = df[df["model"] == "3_between_within"].copy()

PARAM_LABEL = {
    "mesor_blup": "MESOR (bpm)",
    "amplitude_blup": "Amplitude (bpm)",
    "acrophase_blup": "Acrophase (h)",
}
HORM_LABEL = {
    "phs_dhea": "DHEA",
    "phs_testosterone": "Testosterone",
    "phs_estradiol": "Estradiol",
}
HORM_ORDER = ["phs_dhea", "phs_testosterone", "phs_estradiol"]
PARAM_ORDER = ["mesor_blup", "amplitude_blup", "acrophase_blup"]

rows = []
for _, r in bw.iterrows():
    rh = r["rhythm"]; ho = r["hormone"]
    stem = ho
    for kind in ("mean", "dev"):
        b = r[f"b__{stem}_{kind}"]
        se = r[f"se__{stem}_{kind}"]
        p = r[f"p__{stem}_{kind}"]
        if pd.notna(b):
            rows.append({
                "rhythm": rh, "rhythm_label": PARAM_LABEL[rh],
                "hormone": ho, "hormone_label": HORM_LABEL[ho],
                "effect": "between" if kind == "mean" else "within",
                "beta": float(b), "se": float(se),
                "ci_lo": float(b - 1.96 * se), "ci_hi": float(b + 1.96 * se),
                "p": float(p), "n_obs": int(r["n_obs"]), "n_subj": int(r["n_groups"]),
            })

tab = pd.DataFrame(rows)
tab.to_csv(OUT / "between_within_table.tsv", sep="\t", index=False)


plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

BETWEEN = "#888888"
WITHIN = "#D62728"

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
y_positions = np.arange(len(HORM_ORDER))

for ax, param in zip(axes, PARAM_ORDER):
    sub = tab[tab["rhythm"] == param]
    for i, ho in enumerate(HORM_ORDER):
        for offset, kind, color, marker in [(-0.15, "between", BETWEEN, "o"),
                                              (+0.15, "within", WITHIN, "s")]:
            r = sub[(sub["hormone"] == ho) & (sub["effect"] == kind)]
            if r.empty: continue
            r = r.iloc[0]
            y = i + offset
            ax.errorbar(r["beta"], y, xerr=[[r["beta"] - r["ci_lo"]], [r["ci_hi"] - r["beta"]]],
                        fmt=marker, color=color, ecolor=color, capsize=3, markersize=7,
                        label=kind.capitalize() if (i == 0 and ax is axes[0]) else None)
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([HORM_LABEL[h] for h in HORM_ORDER])
    ax.set_xlabel(f"beta per hormone z\n{PARAM_LABEL[param]}")
    ax.invert_yaxis()

axes[0].legend(frameon=False, loc="best")
plt.tight_layout()
plt.savefig(FIG_OUT / "fig_within_person_coupling.png", dpi=300, bbox_inches="tight")
plt.close()


md = [
    "# Paper 2 - Within-person hormone coupling to cosinor rhythm\n\n",
    "Model: rhythm ~ age_c + hormone_mean + hormone_dev + is_female, ",
    "random intercept per participant.\n",
    "- `hormone_mean`: person-level average across waves (between-person trait)\n",
    "- `hormone_dev`:  within-person deviation at each wave (state)\n\n",
    "Units: bpm (MESOR, Amplitude) or hours (Acrophase) per hormone z-score.\n\n",
    "| Rhythm | Hormone | Effect | beta | 95% CI | p | N obs / subj |\n",
    "|---|---|---|---|---|---|---|\n",
]
for _, r in tab.iterrows():
    md.append(
        f"| {r['rhythm_label']} | {r['hormone_label']} | {r['effect']} | "
        f"{r['beta']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['p']:.2g} | "
        f"{r['n_obs']:,} / {r['n_subj']:,} |\n"
    )
md.append(
    "\n## Read-out\n\n"
    "- **MESOR**: testosterone shows a strong within-person signal "
    "(state elevation by 1 SD -> MESOR 0.53 bpm lower; p~1e-8), distinct from "
    "the smaller between-person effect (-0.38 bpm/z). DHEA and estradiol are between-only.\n"
    "- **Amplitude**: hormone effects are entirely between-person; no within-person coupling.\n"
    "- **Acrophase**: testosterone and DHEA both show within-person coupling "
    "(state elevation -> later acrophase; p~5e-7 and ~1e-4 respectively), "
    "on top of between-person effects.\n\n"
    "Interpretation: the rhythm signal is biologically responsive to within-person "
    "hormonal fluctuations, not just a stable trait. Strongest state-level coupling "
    "is testosterone -> MESOR. This anchors the rhythm signal as a dynamic biomarker.\n"
)
(OUT / "summary.md").write_text("".join(md))
print(f"Wrote {OUT}/ and {FIG_OUT}/fig_within_person_coupling.png")
print(f"Rows in table: {len(tab)}")
