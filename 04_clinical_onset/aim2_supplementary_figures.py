"""Build 4 Aim 2 figures with blue/red scheme, minimal text."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
LIGHT_BLUE = "#4393C3"
LIGHT_RED = "#D6604D"
GRAY = "#999999"

OUT = "/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/ABCD/ABCD Actigraphy Resource Paper/Longitudinal Paper/combined paper"


# FIGURE A: Transdiagnostic forest plot
def fig_a():
    outcomes = [
        "Hypertension", "Obesity",
        "ODD", "Conduct", "ADHD",
        "Somatic", "Anxiety", "Depression",
    ]
    domain = ["cardio", "cardio", "ext", "ext", "ext", "int", "int", "int"]
    colors = [RED if d == "cardio" else BLUE for d in domain]

    mesor =     [1.65, 1.39, 1.29, 1.52, 1.50, 1.46, 1.46, 1.38]
    mesor_lo =  [1.42, 1.24, 1.07, 1.24, 1.26, 1.27, 1.26, 1.23]
    mesor_hi =  [1.93, 1.56, 1.55, 1.87, 1.78, 1.69, 1.69, 1.55]

    amp =       [0.85, 0.94, 0.99, 1.02, 1.09, 0.91, 0.98, 0.92]
    amp_lo =    [0.74, 0.84, 0.81, 0.84, 0.93, 0.80, 0.86, 0.83]
    amp_hi =    [0.98, 1.05, 1.20, 1.23, 1.28, 1.04, 1.12, 1.03]

    acro =      [1.24, 1.15, 1.23, 1.43, 1.18, 1.12, 1.14, 1.12]
    acro_lo =   [1.08, 1.03, 1.04, 1.19, 1.01, 0.97, 1.00, 1.00]
    acro_hi =   [1.43, 1.29, 1.46, 1.71, 1.37, 1.29, 1.29, 1.25]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    y = np.arange(len(outcomes))

    for ax, data, lo, hi, title in [
        (axes[0], mesor, mesor_lo, mesor_hi, "Mesor"),
        (axes[1], amp, amp_lo, amp_hi, "Amplitude"),
        (axes[2], acro, acro_lo, acro_hi, "Acrophase"),
    ]:
        xerr_lo = [d - l for d, l in zip(data, lo)]
        xerr_hi = [h - d for d, h in zip(data, hi)]

        for i in range(len(outcomes)):
            ax.errorbar(data[i], y[i], xerr=[[xerr_lo[i]], [xerr_hi[i]]],
                        fmt="o", color=colors[i], markersize=6,
                        capsize=3, linewidth=1.2, markeredgewidth=0)
        ax.axvline(1, color=GRAY, linestyle="--", linewidth=0.7, zorder=0)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("OR per 1-SD")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(outcomes)
    axes[0].invert_yaxis()

    # Domain labels
    from matplotlib.patches import Patch
    axes[2].legend(
        [Patch(facecolor=BLUE), Patch(facecolor=RED)],
        ["Psychopathology", "Cardiometabolic"],
        loc="lower right", fontsize=8, frameon=False,
    )

    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{OUT}/figA_transdiagnostic_forest.png", bbox_inches="tight")
    fig.savefig(f"{OUT}/figA_transdiagnostic_forest.pdf", bbox_inches="tight")
    plt.close()
    print("Figure A saved.")


# FIGURE B: Co-development slope-slope correlations
def fig_b():
    outcomes = [
        "Systolic BP", "BMI", "Somatic",
        "Depression", "ADHD", "Anxiety", "ODD", "Conduct",
    ]
    r_vals = [0.89, 0.36, 0.32, 0.22, 0.20, 0.07, 0.05, -0.00]
    p_vals = [0.000526, 3.02e-14, 0.00345, 0.0218, 0.0355, 0.299, 0.548, 0.982]
    domain = ["cardio", "cardio", "int", "int", "ext", "int", "ext", "ext"]
    colors = [RED if d == "cardio" else BLUE for d in domain]
    sig = [p < 0.05 for p in p_vals]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    y = np.arange(len(outcomes))

    for i in range(len(outcomes)):
        marker = "o" if sig[i] else "o"
        alpha = 1.0 if sig[i] else 0.35
        ax.barh(y[i], r_vals[i], height=0.6, color=colors[i], alpha=alpha,
                edgecolor="none")

    ax.axvline(0, color=GRAY, linestyle="-", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(outcomes)
    ax.set_xlabel("Slope-slope correlation (r)")
    ax.set_title("Co-development with mesor trajectory", fontsize=11,
                 fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(-0.15, 1.0)

    from matplotlib.patches import Patch
    ax.legend(
        [Patch(facecolor=BLUE), Patch(facecolor=RED)],
        ["Psychopathology", "Cardiometabolic"],
        loc="lower right", fontsize=8, frameon=False,
    )

    fig.tight_layout()
    fig.savefig(f"{OUT}/figB_codevelopment.png", bbox_inches="tight")
    fig.savefig(f"{OUT}/figB_codevelopment.pdf", bbox_inches="tight")
    plt.close()
    print("Figure B saved.")


# FIGURE C: Incremental AUC waterfall
def fig_c():
    outcomes = [
        "Depression", "Anxiety", "ADHD", "Somatic",
        "Conduct", "ODD", "Obesity", "Hypertension",
    ]
    # AUC at each step from ultra-clean run
    m1 = [0.640, 0.632, 0.717, 0.643, 0.739, 0.733, 0.662, 0.783]
    m2 = [0.732, 0.753, 0.882, 0.749, 0.835, 0.845, 0.765, 0.841]
    m3 = [0.744, 0.755, 0.885, 0.756, 0.844, 0.854, 0.767, 0.846]
    m4 = [0.764, 0.776, 0.894, 0.766, 0.861, 0.857, 0.781, 0.859]
    domain = ["int", "int", "ext", "int", "ext", "ext", "cardio", "cardio"]
    edge_colors = [BLUE if d != "cardio" else RED for d in domain]

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(outcomes))
    bar_h = 0.65

    # Stack: M1 (light gray), M2-M1 (medium gray), M3-M2 (light blue), M4-M3 (red)
    d_m1 = [v - 0.5 for v in m1]
    d_m2 = [m2[i] - m1[i] for i in range(len(outcomes))]
    d_m3 = [m3[i] - m2[i] for i in range(len(outcomes))]
    d_m4 = [m4[i] - m3[i] for i in range(len(outcomes))]

    ax.barh(y, d_m1, height=bar_h, left=0.5, color="#D9D9D9", label="Demographics")
    ax.barh(y, d_m2, height=bar_h, left=m1, color="#A6CEE3", label="+ Baseline symptoms")
    ax.barh(y, d_m3, height=bar_h, left=m2, color=LIGHT_BLUE, label="+ Behavioral wearables")
    ax.barh(y, d_m4, height=bar_h, left=m3, color=RED, label="+ Cardiac rhythm")

    # Add M4 AUC labels
    for i in range(len(outcomes)):
        ax.text(m4[i] + 0.005, y[i], f".{int(m4[i]*1000):03d}",
                va="center", ha="left", fontsize=8, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(outcomes)
    ax.set_xlabel("AUC")
    ax.set_xlim(0.5, 0.95)
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.set_title("Incremental discrimination for clinical onset",
                 fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(f"{OUT}/figC_incremental_auc.png", bbox_inches="tight")
    fig.savefig(f"{OUT}/figC_incremental_auc.pdf", bbox_inches="tight")
    plt.close()
    print("Figure C saved.")


# FIGURE D: Between-person mesor vs within-person SD acrophase
def fig_d():
    outcomes = [
        "Hypertension", "Obesity",
        "ODD", "Conduct", "ADHD",
        "Somatic", "Anxiety", "Depression",
    ]
    domain = ["cardio", "cardio", "ext", "ext", "ext", "int", "int", "int"]
    colors = [RED if d == "cardio" else BLUE for d in domain]

    # Mesor ORs (between-person, ultra-clean)
    mes_or =  [1.65, 1.39, 1.29, 1.52, 1.50, 1.46, 1.46, 1.38]
    mes_lo =  [1.42, 1.24, 1.07, 1.24, 1.26, 1.27, 1.26, 1.23]
    mes_hi =  [1.93, 1.56, 1.55, 1.87, 1.78, 1.69, 1.69, 1.55]

    # SD acrophase ORs (within-person, ultra-clean)
    sda_or =  [1.29, 1.23, 1.39, 1.33, 1.17, 1.33, 1.24, 1.32]
    sda_lo =  [1.11, 1.10, 1.18, 1.10, 1.01, 1.17, 1.10, 1.18]
    sda_hi =  [1.49, 1.39, 1.64, 1.59, 1.36, 1.51, 1.41, 1.49]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    y = np.arange(len(outcomes))

    # Panel A: Mesor
    for i in range(len(outcomes)):
        ax1.errorbar(mes_or[i], y[i],
                     xerr=[[mes_or[i]-mes_lo[i]], [mes_hi[i]-mes_or[i]]],
                     fmt="o", color=colors[i], markersize=6,
                     capsize=3, linewidth=1.2, markeredgewidth=0)
    ax1.axvline(1, color=GRAY, linestyle="--", linewidth=0.7, zorder=0)
    ax1.set_title("Typical-day mesor", fontsize=11, fontweight="bold")
    ax1.set_xlabel("OR per 1-SD")
    ax1.set_yticks(y)
    ax1.set_yticklabels(outcomes)
    ax1.invert_yaxis()

    # Panel B: SD acrophase
    for i in range(len(outcomes)):
        ax2.errorbar(sda_or[i], y[i],
                     xerr=[[sda_or[i]-sda_lo[i]], [sda_hi[i]-sda_or[i]]],
                     fmt="s", color=colors[i], markersize=6,
                     capsize=3, linewidth=1.2, markeredgewidth=0)
    ax2.axvline(1, color=GRAY, linestyle="--", linewidth=0.7, zorder=0)
    ax2.set_title("Day-to-day acrophase variability", fontsize=11,
                  fontweight="bold")
    ax2.set_xlabel("OR per 1-SD")

    from matplotlib.patches import Patch
    ax2.legend(
        [Patch(facecolor=BLUE), Patch(facecolor=RED)],
        ["Psychopathology", "Cardiometabolic"],
        loc="lower right", fontsize=8, frameon=False,
    )

    fig.tight_layout(w_pad=2)
    fig.savefig(f"{OUT}/figD_mesor_vs_sdacro.png", bbox_inches="tight")
    fig.savefig(f"{OUT}/figD_mesor_vs_sdacro.pdf", bbox_inches="tight")
    plt.close()
    print("Figure D saved.")


if __name__ == "__main__":
    fig_a()
    fig_b()
    fig_c()
    fig_d()
    print("\nAll figures saved.")
