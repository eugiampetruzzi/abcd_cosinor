"""Figure: Cosinor rhythm at Wave 2.

5-panel layout (single high-quality figure):
    A. Population 24-h HR profile with cosinor fit + interquartile band.
    B. Mesor BLUP distribution (histogram + KDE).
    C. Amplitude BLUP distribution.
    D. Acrophase BLUP distribution (polar / clock-style).
    E. Construct validity: cosinor mesor vs concurrent BP-cuff resting HR.

Style: Arial; colorful but minimal text. Caption carries the detail.
Saves PNG (300 dpi) + SVG to derivatives/figures_paper2/cosinor/.
"""
from __future__ import annotations
import sys
from pathlib import Path
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patches as mpatches
from scipy.stats import circmean, circstd, gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import DERIV, ONEDRIVE_OUT  # noqa: E402

# Style
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#222222",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})
COL = {
    "blue":   "#1F4E79",
    "teal":   "#2BA89A",
    "coral":  "#E8635D",
    "amber":  "#F2A93B",
    "purple": "#6B5B95",
    "lavender":"#A4B0DA",
    "grey":   "#7A7A7A",
    "lightgrey": "#D9D9D9",
}

W2 = "ses-02A"

OUT = DERIV / "figures_paper2" / "cosinor"
OUT.mkdir(parents=True, exist_ok=True)


# Load
print("Loading ...")
bl = (pl.read_parquet(DERIV/f"cosinor_features/per_wave/{W2}/participant_blups.parquet")
        .rename({"subject_id":"participant_id"}).to_pandas())
hp = pl.read_parquet(DERIV/f"hourly_profiles/{W2}.parquet").to_pandas()
ph = pl.read_parquet(ONEDRIVE_OUT/"outcomes/master_outcomes_physical_health.parquet").to_pandas()
phw2 = ph[ph["session_id"]==W2][["participant_id","bp_hrate_mean"]]


# Panel A: Population 24-h HR rhythm
print("Building Panel A: population HR rhythm ...")
profiles = (hp.groupby("clock_hour")
              .agg(p10=("hr_median", lambda x: np.nanpercentile(x, 10)),
                   p25=("hr_median", lambda x: np.nanpercentile(x, 25)),
                   p50=("hr_median", "median"),
                   p75=("hr_median", lambda x: np.nanpercentile(x, 75)),
                   p90=("hr_median", lambda x: np.nanpercentile(x, 90)))
              .reset_index())

# Population cosinor fit (using mean BLUP parameters)
pop_mesor = bl["mesor_blup"].mean()
pop_amp   = bl["amplitude_blup"].mean()
acro_rad = (bl["acrophase_blup"].values / 24) * 2*np.pi
pop_acro_circmean = (circmean(acro_rad) * 24/(2*np.pi)) % 24
hours_grid = np.linspace(0, 24, 200)
pop_curve = pop_mesor + pop_amp * np.cos(2*np.pi*(hours_grid - pop_acro_circmean)/24)


# Pre-compute distributions
mesor = bl["mesor_blup"].dropna().values
ampl  = bl["amplitude_blup"].dropna().values
acro  = bl["acrophase_blup"].dropna().values

# For Panel D polar
acro_rad_full = (acro / 24) * 2*np.pi
acro_circ_mean = (circmean(acro_rad_full) * 24/(2*np.pi)) % 24
acro_circ_sd   = circstd(acro_rad_full) * 24/(2*np.pi)


# Panel E: Construct validity
m_bp = bl.merge(phw2, on="participant_id", how="inner").dropna(subset=["mesor_blup","bp_hrate_mean"])
r_val = m_bp["mesor_blup"].corr(m_bp["bp_hrate_mean"])


# Build figure
print("Drawing ...")
fig = plt.figure(figsize=(11, 7.2), constrained_layout=False)
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.85],
                      hspace=0.55, wspace=0.45,
                      left=0.07, right=0.97, top=0.93, bottom=0.10)

# Panel A  Population HR rhythm (top, spans 2 cols)
axA = fig.add_subplot(gs[0, 0:2])
axA.fill_between(profiles["clock_hour"]+0.5, profiles["p10"], profiles["p90"],
                 color=COL["lavender"], alpha=0.45, linewidth=0,
                 label="10–90%ile")
axA.fill_between(profiles["clock_hour"]+0.5, profiles["p25"], profiles["p75"],
                 color=COL["purple"], alpha=0.45, linewidth=0,
                 label="25–75%ile")
axA.plot(profiles["clock_hour"]+0.5, profiles["p50"], color=COL["blue"],
         lw=2.6, label="Median")
axA.plot(hours_grid, pop_curve, color=COL["coral"], lw=2.4, ls="--",
         label="Cosinor fit")
axA.set_xlim(0, 24); axA.set_xticks(np.arange(0, 25, 4))
ymin, ymax = profiles["p10"].min(), profiles["p90"].max()
yrange = ymax - ymin
axA.set_ylim(ymin - 0.10*yrange, ymax + 0.30*yrange)  # extra headroom
axA.set_xlabel("Clock hour")
axA.set_ylabel("Heart rate (bpm)")
axA.set_title("A.  Population 24-hour HR rhythm at Wave 2", loc="left",
              fontweight="bold", color=COL["blue"])
axA.grid(alpha=0.18, lw=0.6)
axA.legend(loc="upper left", bbox_to_anchor=(0.01, 0.97),
           frameon=False, fontsize=8.5, handlelength=1.4,
           ncols=2, columnspacing=1.0, handletextpad=0.5)

# Panel B  Mesor distribution
axB = fig.add_subplot(gs[0, 2])
counts_b, bins_b, _ = axB.hist(mesor, bins=45, color=COL["blue"], alpha=0.7,
                                 edgecolor="white", linewidth=0.4)
kde = gaussian_kde(mesor)
xx = np.linspace(mesor.min(), mesor.max(), 300)
ax2 = axB.twinx()
ax2.plot(xx, kde(xx), color=COL["coral"], lw=2.0)
ax2.set_yticks([]); ax2.spines["right"].set_visible(False); ax2.spines["top"].set_visible(False)
axB.axvline(np.median(mesor), color=COL["coral"], lw=1.5, ls=":")
axB.set_ylim(0, counts_b.max()*1.10)
axB.set_xlabel("Mesor (bpm)")
axB.set_ylabel("Participants")
axB.set_title(f"B.  Mesor   (median = {np.median(mesor):.1f} bpm)",
              loc="left", fontweight="bold", color=COL["blue"])

# Panel C  Amplitude distribution
axC = fig.add_subplot(gs[0, 3])
counts_c, bins_c, _ = axC.hist(ampl, bins=45, color=COL["teal"], alpha=0.78,
                                 edgecolor="white", linewidth=0.4)
kde2 = gaussian_kde(ampl)
xx2 = np.linspace(ampl.min(), ampl.max(), 300)
ax2 = axC.twinx(); ax2.plot(xx2, kde2(xx2), color=COL["coral"], lw=2.0)
ax2.set_yticks([]); ax2.spines["right"].set_visible(False); ax2.spines["top"].set_visible(False)
axC.axvline(np.median(ampl), color=COL["coral"], lw=1.5, ls=":")
axC.set_ylim(0, counts_c.max()*1.10)
axC.set_xlabel("Amplitude (bpm)")
axC.set_ylabel("Participants")
axC.set_title(f"C.  Amplitude   (median = {np.median(ampl):.1f} bpm)",
              loc="left", fontweight="bold", color=COL["blue"])

# Panel D  Acrophase polar (clock-style)
axD = fig.add_subplot(gs[1, 0:2], projection="polar")
# Convert hours [0..24) to radians; clock starts at top (12 o'clock = 0 hr)
# Standard astronomic: angle = 2*pi * hour / 24, but we want 0h at top going CW
theta = (acro / 24) * 2*np.pi
# Polar histogram
n_bins = 48
bins = np.linspace(0, 2*np.pi, n_bins+1)
counts, edges = np.histogram(theta, bins=bins)
widths = np.diff(edges)
centers = (edges[:-1] + edges[1:]) / 2

axD.set_theta_zero_location("N")  # 0 at top
axD.set_theta_direction(-1)        # clockwise
bars = axD.bar(centers, counts, width=widths*0.95,
               color=COL["amber"], alpha=0.78, edgecolor="white", linewidth=0.4)
# Mark circular mean
mean_theta = (acro_circ_mean / 24) * 2*np.pi
axD.plot([mean_theta, mean_theta], [0, max(counts)*1.1], color=COL["coral"],
         lw=2.4, label=f"Circular mean = {acro_circ_mean:.1f} h")
# Hour ticks at 0, 6, 12, 18
hour_labels = ["00", "06", "12", "18"]
hour_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
axD.set_xticks(hour_angles)
axD.set_xticklabels(hour_labels, fontsize=9)
axD.set_yticklabels([])
axD.set_title("D.  Acrophase (clock hour of HR peak)", loc="left",
              fontweight="bold", color=COL["blue"], pad=14)
axD.legend(loc=(0.78, -0.05), frameon=False, fontsize=8.5)

# Panel E  Construct validity
axE = fig.add_subplot(gs[1, 2:4])
# Hexbin scatter (colorful + dense points)
hb = axE.hexbin(m_bp["mesor_blup"], m_bp["bp_hrate_mean"], gridsize=40,
                cmap="viridis", mincnt=1, alpha=0.95, linewidths=0)
# Trend line
slope, intercept = np.polyfit(m_bp["mesor_blup"], m_bp["bp_hrate_mean"], 1)
xt = np.linspace(m_bp["mesor_blup"].min(), m_bp["mesor_blup"].max(), 200)
axE.plot(xt, slope*xt+intercept, color=COL["coral"], lw=2.4,
         label=f"r = {r_val:.2f}  (N = {len(m_bp):,})")
axE.set_xlabel("Cosinor mesor (bpm)")
axE.set_ylabel("BP-cuff resting HR (bpm)")
axE.set_title("E.  Construct validity at Wave 2", loc="left",
              fontweight="bold", color=COL["blue"])
axE.legend(loc="upper left", frameon=False, fontsize=9)
cb = fig.colorbar(hb, ax=axE, fraction=0.04, pad=0.02)
cb.set_label("N participants", fontsize=8.5)
cb.ax.tick_params(labelsize=8)

# Save
png = OUT / "fig_cosinor_W2.png"
svg = OUT / "fig_cosinor_W2.svg"
pdf = OUT / "fig_cosinor_W2.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(svg,            bbox_inches="tight")
fig.savefig(pdf,            bbox_inches="tight")
print(f"\nWrote:\n  {png}\n  {svg}\n  {pdf}")
