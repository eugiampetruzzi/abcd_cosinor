"""Figure 6 — Conditional prediction across the depression-obesity axis.

Two-panel forest plot showing that Wave-2 cosinor parameters prospectively
predict onset of a second condition in both directions of the depression -
obesity axis, but via dissociable rhythm features:

  Panel A  Depression by Wave 2 -> incident obesity at W3 or W4.
  Panel B  Obesity by Wave 2 -> incident depression at W3 or W4.

Each panel includes a small inset timeline above the forest, showing
W1/W2 (anchor measurement) -> W2 (cosinor measurement) -> W3/W4 (onset).

Inputs:
    results/tables/conditional_prediction_primary.tsv
"""
from __future__ import annotations
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import RESULTS_DIR, FIGS_DIR  # noqa: E402

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

C_HL    = "#E8635D"   # coral — highlighted predictor / cosinor measurement
C_GRY   = "#7A7A7A"
C_TITLE = "#1F4E79"   # navy
C_BAND  = "#F0F0F0"
C_ANCH  = "#1F4E79"   # navy for anchor block on timeline
C_OUT   = "#3E8E7E"   # teal-green for outcome block on timeline

PRED_ORDER = [
    ("mesor",              "Mesor"),
    ("amplitude",          "Amplitude"),
    ("acrophase",          "Acrophase"),
    ("SD daily mesor",     "SD daily mesor"),
    ("SD daily amplitude", "SD daily amplitude"),
    ("SD daily acrophase", "SD daily acrophase"),
]


def _load_cell(df: pd.DataFrame, anchor: str, target: str) -> pd.DataFrame:
    sub = df[(df["anchor_condition"] == anchor)
              & (df["target_condition"] == target)].copy()
    sub = sub.set_index("predictor").reindex([p for p, _ in PRED_ORDER])
    sub.index.name = "predictor"
    return sub.reset_index()


def _draw_timeline(ax, *, anchor_label: str, target_label: str):
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.8, 1.6)
    # Backbone wave line
    ax.plot([0.3, 3.7], [0, 0], color="#888888", lw=1.0, zorder=1)
    # Wave markers + labels (W2 highlighted in coral as the cosinor measurement)
    for x, label in [(0.5, "W1"), (1.5, "W2"), (2.5, "W3"), (3.5, "W4")]:
        col = C_HL if label == "W2" else "#444444"
        size = 8 if label == "W2" else 5
        ax.plot(x, 0, "o", color=col, markersize=size, zorder=3,
                  markeredgecolor="white", markeredgewidth=0.8)
        if label == "W2":
            ax.text(x, -0.4, "W2  (cosinor)", ha="center", va="top",
                      fontsize=9, color=C_HL, fontweight="bold")
        else:
            ax.text(x, -0.4, label, ha="center", va="top",
                      fontsize=9, color="#444444")
    # Anchor block (W1 - W2)
    ax.plot([0.5, 1.5], [0.85, 0.85], color=C_ANCH, lw=2.4, zorder=2)
    ax.plot([0.5, 0.5], [0.7, 0.85], color=C_ANCH, lw=1.2, zorder=2)
    ax.plot([1.5, 1.5], [0.7, 0.85], color=C_ANCH, lw=1.2, zorder=2)
    ax.text(1.0, 1.05, f"Anchor: {anchor_label}", ha="center", va="bottom",
              fontsize=9, color=C_ANCH, fontweight="bold")
    # Outcome block (W3 - W4)
    ax.plot([2.5, 3.5], [0.85, 0.85], color=C_OUT, lw=2.4, zorder=2)
    ax.plot([2.5, 2.5], [0.7, 0.85], color=C_OUT, lw=1.2, zorder=2)
    ax.plot([3.5, 3.5], [0.7, 0.85], color=C_OUT, lw=1.2, zorder=2)
    ax.text(3.0, 1.05, f"Onset: {target_label}", ha="center", va="bottom",
              fontsize=9, color=C_OUT, fontweight="bold")
    ax.set_axis_off()


def _draw_panel(ax, cell_df: pd.DataFrame, *, hl_predictor: str,
                  panel_label: str):
    n_rows = len(cell_df)
    y = np.arange(n_rows)[::-1]
    ax.axhspan(y[-1] - 0.5, y[3] + 0.5, color=C_BAND, alpha=0.55, zorder=0)
    ax.axvline(1.0, color="#444444", lw=0.9, ls="--", zorder=1)
    for i, (_, r) in enumerate(cell_df.iterrows()):
        is_hl = (r["predictor"] == hl_predictor)
        col = C_HL if is_hl else C_GRY
        lw = 2.4 if is_hl else 1.6
        ms = 9 if is_hl else 6
        zorder = 4 if is_hl else 2
        if pd.isna(r["or_per_sd"]):
            continue
        ax.errorbar(r["or_per_sd"], y[i],
                      xerr=[[r["or_per_sd"] - r["ci_lo"]],
                             [r["ci_hi"] - r["or_per_sd"]]],
                      fmt="o", color=col, ecolor=col, elinewidth=lw,
                      capsize=4, markersize=ms,
                      markeredgecolor="white", markeredgewidth=1.0,
                      zorder=zorder)
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in PRED_ORDER])
    for tick, (pid, _) in zip(ax.get_yticklabels(), PRED_ORDER):
        if pid == hl_predictor:
            tick.set_fontweight("bold")
            tick.set_color(C_HL)
    ax.set_xscale("log")
    ax.set_xlim(0.45, 3.5)
    ax.xaxis.set_major_locator(mticker.FixedLocator(
        [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(
        ["0.5", "0.75", "1.0", "1.5", "2.0", "3.0"]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("Odds ratio (per 1-SD predictor)")
    ax.set_title(panel_label, loc="left", fontweight="bold",
                  color=C_TITLE, pad=6)
    ax.set_ylim(y[-1] - 0.6, y[0] + 0.6)
    ax.grid(axis="x", alpha=0.18, lw=0.6)


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "tables"
                       / "conditional_prediction_primary.tsv", sep="\t")

    cell_dep_to_ob = _load_cell(df, "depression", "obesity")
    cell_ob_to_dep = _load_cell(df, "obesity", "depression")

    fig = plt.figure(figsize=(12.5, 5.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 4.0],
                            hspace=0.30, wspace=0.45)
    ax_tlA = fig.add_subplot(gs[0, 0])
    ax_tlB = fig.add_subplot(gs[0, 1])
    axA = fig.add_subplot(gs[1, 0])
    axB = fig.add_subplot(gs[1, 1])

    _draw_timeline(ax_tlA, anchor_label="Depression",
                     target_label="Obesity")
    _draw_timeline(ax_tlB, anchor_label="Obesity",
                     target_label="Depression")

    _draw_panel(axA, cell_dep_to_ob,
                  hl_predictor="mesor",
                  panel_label="A.  Depression → obesity")
    _draw_panel(axB, cell_ob_to_dep,
                  hl_predictor="SD daily mesor",
                  panel_label="B.  Obesity → depression")

    out_dir = FIGS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"fig6_conditional_prediction.{ext}",
                     bbox_inches="tight")
    print(f"Wrote {out_dir}/fig6_conditional_prediction.{{png,svg,pdf}}")


if __name__ == "__main__":
    main()
