"""Figure 2 — Population 24-hour HR rhythm at Wave 2.

Single-panel figure. Five percentile lines (10/25/50/75/90 of the
across-participant distribution of median HR at each clock hour) plus the
population-mean cosinor fit overlaid.

Population-mean cosinor fit:
    pop_mesor   = arithmetic mean of per-participant mesor BLUPs
    pop_amp     = arithmetic mean of per-participant amplitude BLUPs
    pop_acro    = circular mean of per-participant acrophase BLUPs
    fit(t)      = pop_mesor + pop_amp · cos(2π(t − pop_acro)/24)

Outputs:
    figures/fig2_population_rhythm.{png,svg,pdf}
"""
from __future__ import annotations
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import circmean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import (                          # noqa: E402
    COSINOR_BLUP_W2, HOURLY_PROFILE_W2, FIGS_DIR,
)

FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ----- Style -----
mpl.rcParams.update({
    "font.family": "Arial", "font.sans-serif": ["Arial"],
    "font.size": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.linewidth": 0.9, "axes.edgecolor": "#222222",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

# Colours: blue family for percentiles, coral for cosinor fit
C_OUTER = "#A4B0DA"   # 10th / 90th
C_MID   = "#4A6FAF"   # 25th / 75th
C_MED   = "#1F4E79"   # 50th
C_FIT   = "#E8635D"   # population cosinor fit


def main() -> None:
    bl = pl.read_parquet(COSINOR_BLUP_W2).to_pandas()
    bl = bl.dropna(subset=["mesor_blup", "amplitude_blup", "acrophase_blup"])
    hp = pl.read_parquet(HOURLY_PROFILE_W2).to_pandas()
    n = len(bl)

    # Per-clock-hour percentiles across participants
    prof = (hp.groupby("clock_hour")
              .agg(p10=("hr_median", lambda x: np.nanpercentile(x, 10)),
                   p25=("hr_median", lambda x: np.nanpercentile(x, 25)),
                   p50=("hr_median", "median"),
                   p75=("hr_median", lambda x: np.nanpercentile(x, 75)),
                   p90=("hr_median", lambda x: np.nanpercentile(x, 90)))
              .reset_index())
    x = prof["clock_hour"].values + 0.5  # centred on hour midpoint

    # Population-mean cosinor curve
    pop_mesor = float(bl["mesor_blup"].mean())
    pop_amp   = float(bl["amplitude_blup"].mean())
    acro_rad  = (bl["acrophase_blup"].to_numpy() / 24) * 2 * np.pi
    pop_acro  = float((circmean(acro_rad) * 24 / (2 * np.pi)) % 24)
    hours_grid = np.linspace(0, 24, 400)
    pop_curve = pop_mesor + pop_amp * np.cos(2 * np.pi * (hours_grid - pop_acro) / 24)
    print(f"  N = {n:,}")
    print(f"  Pop mesor = {pop_mesor:.2f} bpm  amp = {pop_amp:.2f} bpm  "
          f"acrophase = {pop_acro:.2f} hr")

    # ----- Draw -----
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(x, prof["p10"], color=C_OUTER, lw=1.1, label="10th / 90th percentile")
    ax.plot(x, prof["p90"], color=C_OUTER, lw=1.1)
    ax.plot(x, prof["p25"], color=C_MID,   lw=1.6, label="25th / 75th percentile")
    ax.plot(x, prof["p75"], color=C_MID,   lw=1.6)
    ax.plot(x, prof["p50"], color=C_MED,   lw=2.6, label="Median")
    ax.plot(hours_grid, pop_curve, color=C_FIT, lw=2.2, ls="--",
             label="Population cosinor fit")

    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.set_xlabel("Clock hour")
    ax.set_ylabel("Heart rate (bpm)")
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5,
                handlelength=1.8, labelspacing=0.4)
    fig.tight_layout()

    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIGS_DIR / f"fig2_population_rhythm.{ext}",
                     bbox_inches="tight")
    print(f"Wrote {FIGS_DIR}/fig2_population_rhythm.{{png,svg,pdf}}")


if __name__ == "__main__":
    main()
