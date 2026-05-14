"""Shared 4-panel figure builder for case-vs-control onset analyses.

Used by `fig3_depression.py`, `fig4_obesity.py`, and `fig5_hypertension.py`.
Each script supplies its analytic frame + cohort labels; this module owns the
layout, styling, and panel logic.

Layout:
    A (top-left)     between-person cosinor parameters (z, cases vs ctrl ± 95 % CI)
    B (top-right)    24-h HR rhythm — cases vs ctrl, mean ± 95 % CI
    C (bottom-left)  within-person cosinor SDs (z, cases vs ctrl ± 95 % CI)
    D (bottom-right) hour-by-hour difference (cases − ctrl) ± 95 % CI
"""
from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.lines import Line2D

from .modeling import fit_logistic_cluster
from .paths import HOURLY_PROFILE_W2

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
C_CTRL = "#1F4E79"   # navy
C_CASE = "#E8635D"   # coral
C_DIFF = "#6B5B95"   # purple
GREY   = "#7A7A7A"

BETWEEN = [("typical_day_mesor",     "Mesor"),
            ("typical_day_amplitude", "Amplitude"),
            ("typical_day_acrophase", "Acrophase")]
WITHIN = [("SD_daily_mesor",      "SD daily\nmesor"),
          ("SD_daily_amplitude",  "SD daily\namplitude"),
          ("SD_daily_acrophase",  "SD daily\nacrophase")]


def stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def _param_means(coh: pd.DataFrame, params) -> list[dict]:
    coh = coh.dropna(subset=[c for c, _ in params]).copy()
    for col, _ in params:
        coh[f"{col}_z"] = (coh[col] - coh[col].mean()) / coh[col].std()
    bars = []
    for col, label in params:
        z = f"{col}_z"
        g_c = coh.loc[coh["onset"] == 0, z]
        g_k = coh.loc[coh["onset"] == 1, z]
        r = fit_logistic_cluster(coh, [col], return_predictor=col)
        bars.append({
            "label": label,
            "ctrl_mean": float(g_c.mean()), "case_mean": float(g_k.mean()),
            "ctrl_ci": float(1.96 * g_c.std(ddof=1) / np.sqrt(len(g_c))),
            "case_ci": float(1.96 * g_k.std(ddof=1) / np.sqrt(len(g_k))),
            "p": r.p, "OR": r.OR,
            "n_ctrl": int((coh["onset"] == 0).sum()),
            "n_case": int((coh["onset"] == 1).sum()),
        })
    return bars


def _draw_param_panel(ax, bars, title):
    x_pos = np.arange(len(bars)); offset = 0.18
    ctrl_means = [b["ctrl_mean"] for b in bars]
    case_means = [b["case_mean"] for b in bars]
    ctrl_err   = [b["ctrl_ci"]   for b in bars]
    case_err   = [b["case_ci"]   for b in bars]
    ax.axhline(0.0, color=GREY, lw=0.9)
    ax.errorbar(x_pos - offset, ctrl_means, yerr=ctrl_err, fmt="o",
                  color=C_CTRL, ecolor=C_CTRL, elinewidth=2.0, capsize=4,
                  markersize=10, markeredgecolor="white", markeredgewidth=1.0)
    ax.errorbar(x_pos + offset, case_means, yerr=case_err, fmt="o",
                  color=C_CASE, ecolor=C_CASE, elinewidth=2.0, capsize=4,
                  markersize=10, markeredgecolor="white", markeredgewidth=1.0)
    ymax = max(max(ctrl_means) + max(ctrl_err),
               max(case_means) + max(case_err))
    ymin = min(min(ctrl_means) - max(ctrl_err),
               min(case_means) - max(case_err))
    yrange = ymax - ymin
    for i, b in enumerate(bars):
        s = stars(b["p"])
        if not s: continue
        y = max(b["ctrl_mean"] + b["ctrl_ci"],
                  b["case_mean"] + b["case_ci"]) + 0.08 * yrange
        ax.plot([x_pos[i] - offset, x_pos[i] + offset], [y, y],
                  color="#222222", lw=1.0)
        ax.text(x_pos[i], y + 0.01 * yrange, s, ha="center", va="bottom",
                  fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([b["label"] for b in bars])
    ax.set_ylabel("Standardized score (z)")
    ax.set_ylim(ymin - 0.20 * yrange, ymax + 0.30 * yrange)
    ax.grid(axis="y", alpha=0.18, lw=0.6)
    ax.set_title(title, loc="left", fontweight="bold", color=C_CTRL, pad=6)


def build_combined_figure(
    analytic_df: pd.DataFrame,
    *,
    case_label: str,
    control_label: str,
    out_dir: Path,
    out_stem: str,
    panel_a_title: str = "A.  Between-person cosinor features",
    panel_c_title: str = "C.  Within-person cosinor features",
    panel_b_title: str = "B.  24-hour HR rhythm at Wave 2",
    panel_d_title: str = "D.  Hour-by-hour difference",
):
    """Render the 4-panel case-vs-control combined figure for one outcome.

    `analytic_df` must already contain:
        participant_id, onset (0/1),
        typical_day_mesor / amplitude / acrophase,
        SD_daily_mesor / amplitude / acrophase
    """
    df = analytic_df
    n = len(df); n_case = int(df["onset"].sum()); n_ctrl = n - n_case

    bars_btw = _param_means(df, BETWEEN)
    bars_wpu = _param_means(df, WITHIN)
    n_wp = bars_wpu[0]["n_ctrl"] + bars_wpu[0]["n_case"]
    n_wp_case = bars_wpu[0]["n_case"]

    # 24-h rhythm + hourly difference
    hp = pl.read_parquet(HOURLY_PROFILE_W2).to_pandas()
    hp = hp.merge(df[["participant_id", "onset"]],
                    left_on="subject_id", right_on="participant_id", how="inner")
    agg = (hp.groupby(["onset", "clock_hour"])["hr_median"]
              .agg(["mean", "std", "count"]).reset_index())
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    agg["ci_lo"] = agg["mean"] - 1.96 * agg["se"]
    agg["ci_hi"] = agg["mean"] + 1.96 * agg["se"]
    ctrl_p = agg[agg["onset"] == 0].sort_values("clock_hour").reset_index(drop=True)
    case_p = agg[agg["onset"] == 1].sort_values("clock_hour").reset_index(drop=True)
    diff = pd.DataFrame({"clock_hour": ctrl_p["clock_hour"].values})
    diff["d_mean"] = case_p["mean"].values - ctrl_p["mean"].values
    diff["d_se"]   = np.sqrt(case_p["se"].values ** 2 + ctrl_p["se"].values ** 2)
    diff["d_lo"]   = diff["d_mean"] - 1.96 * diff["d_se"]
    diff["d_hi"]   = diff["d_mean"] + 1.96 * diff["d_se"]

    # Draw
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4),
                              gridspec_kw={"width_ratios": [1.0, 1.4],
                                            "height_ratios": [1.0, 1.0],
                                            "hspace": 0.40, "wspace": 0.30})
    axA, axB = axes[0]; axC, axD = axes[1]

    _draw_param_panel(axA, bars_btw, panel_a_title)
    axA.text(0.99, 0.97, f"n = {n:,}; cases = {n_case}",
              transform=axA.transAxes, ha="right", va="top",
              fontsize=9, color="#555555")
    _draw_param_panel(axC, bars_wpu, panel_c_title)
    axC.text(0.99, 0.97, f"n = {n_wp:,}; cases = {n_wp_case}",
              transform=axC.transAxes, ha="right", va="top",
              fontsize=9, color="#555555")

    xCv = ctrl_p["clock_hour"].values + 0.5
    xKv = case_p["clock_hour"].values + 0.5
    axB.fill_between(xCv, ctrl_p["ci_lo"], ctrl_p["ci_hi"],
                       color=C_CTRL, alpha=0.18, linewidth=0)
    axB.fill_between(xKv, case_p["ci_lo"], case_p["ci_hi"],
                       color=C_CASE, alpha=0.22, linewidth=0)
    axB.plot(xCv, ctrl_p["mean"], color=C_CTRL, lw=2.2)
    axB.plot(xKv, case_p["mean"], color=C_CASE, lw=2.2)
    axB.set_ylabel("Heart rate (bpm)")
    axB.set_xlim(0, 24); axB.set_xticks(np.arange(0, 25, 4))
    axB.grid(alpha=0.18, lw=0.6)
    axB.set_title(panel_b_title, loc="left",
                   fontweight="bold", color=C_CTRL, pad=6)
    plt.setp(axB.get_xticklabels(), visible=False)

    xD = diff["clock_hour"].values + 0.5
    axD.axhline(0.0, color=GREY, lw=0.9, ls="--")
    axD.fill_between(xD, diff["d_lo"], diff["d_hi"],
                       color=C_DIFF, alpha=0.22, linewidth=0)
    axD.plot(xD, diff["d_mean"], color=C_DIFF, lw=2.0)
    axD.set_xlabel("Clock hour")
    axD.set_ylabel("Δ HR (bpm)\ncases − controls")
    axD.set_xlim(0, 24); axD.set_xticks(np.arange(0, 25, 4))
    axD.grid(alpha=0.18, lw=0.6)
    axD.set_title(panel_d_title, loc="left",
                   fontweight="bold", color=C_CTRL, pad=6)

    handles = [
        Line2D([0], [0], marker="o", color=C_CTRL, lw=0,
               mfc=C_CTRL, mec="white", mew=1.0, markersize=10,
               label=f"{control_label} (n = {n_ctrl:,})"),
        Line2D([0], [0], marker="o", color=C_CASE, lw=0,
               mfc=C_CASE, mec="white", mew=1.0, markersize=10,
               label=f"{case_label} (n = {n_case:,})"),
    ]
    fig.legend(handles=handles,
                loc="lower center", bbox_to_anchor=(0.5, -0.01),
                frameon=False, fontsize=10, ncols=2,
                handlelength=1.8, columnspacing=2.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"{out_stem}.{ext}", bbox_inches="tight")
    print(f"  Wrote {out_dir}/{out_stem}.{{png,svg,pdf}}")
    return {"between": bars_btw, "within": bars_wpu,
             "n": n, "n_case": n_case, "n_wp": n_wp, "n_wp_case": n_wp_case}
