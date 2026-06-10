"""Stage 3a Task 5 - QC plots for cosinor BLUPs.

Reads:
    derivatives/cosinor_features/per_wave/{wave}/participant_blups.parquet
    derivatives/cosinor_features/per_wave/{wave}/population_estimates.json
    derivatives/cosinor_features/pooled/participant_blups.parquet
    derivatives/hourly_profiles/{wave}.parquet (for population-rhythm panels)

Writes to OneDrive qc/stage3_cosinor/:
    fig_population_rhythm_per_wave.png
    fig_blup_distributions.png
    fig_blup_per_wave_vs_pooled_scatter.png
    fig_acrophase_polar.png
    fig_r2_distribution.png
    summary.json    -- structured summary for the chat printout
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import QC, DERIV  # noqa: E402

WAVES = ["ses-02A", "ses-04A", "ses-06A"]
COS = DERIV / "cosinor_features"
PROF = DERIV / "hourly_profiles"
OUT = QC / "stage3_cosinor"
OUT.mkdir(parents=True, exist_ok=True)


def load_per_wave():
    blups = {w: pl.read_parquet(COS / "per_wave" / w / "participant_blups.parquet") for w in WAVES}
    pops = {w: json.loads((COS / "per_wave" / w / "population_estimates.json").read_text()) for w in WAVES}
    diag = {w: json.loads((COS / "per_wave" / w / "model_diagnostics.json").read_text()) for w in WAVES}
    return blups, pops, diag


def load_pooled():
    b = pl.read_parquet(COS / "pooled" / "participant_blups.parquet")
    p = json.loads((COS / "pooled" / "population_estimates.json").read_text())
    d = json.loads((COS / "pooled" / "model_diagnostics.json").read_text())
    return b, p, d


def fig_population_rhythm(blups, pops):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for j, w in enumerate(WAVES):
        ax = axes[j]
        prof = pl.read_parquet(PROF / f"{w}.parquet").to_pandas()
        agg = prof.groupby("clock_hour")["hr_median"].agg(["mean", "median"])
        ax.plot(agg.index, agg["mean"], "o-", color="#4c78a8", lw=1.0, label="population mean")
        ax.fill_between(agg.index,
                        prof.groupby("clock_hour")["hr_median"].quantile(0.25),
                        prof.groupby("clock_hour")["hr_median"].quantile(0.75),
                        color="#4c78a8", alpha=0.15, label="IQR")
        # Cosinor fit overlay
        m = pops[w]["fixed_effects"]["mesor"]["estimate"]
        a = pops[w]["fixed_effects"]["amplitude"]["estimate"]
        phi = pops[w]["fixed_effects"]["acrophase"]["estimate_hours"]
        t = np.linspace(0, 23, 200)
        ax.plot(t, m + a * np.cos(2*np.pi*(t - phi)/24), "-", color="#d62728", lw=1.6, label="cosinor fit")
        ax.set_title(f"{w}\nN={pops[w]['n_subjects']}, mesor={m:.1f}, amp={a:.2f}, acro={phi:.1f}h")
        ax.set_xlabel("clock hour"); ax.set_xticks(range(0, 24, 2)); ax.set_xlim(-0.5, 23.5)
        if j == 0: ax.set_ylabel("hr_continuous (bpm)")
        if j == 2: ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_population_rhythm_per_wave.png", dpi=120); plt.close(fig)


def fig_blup_distributions(blups):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for col, param in enumerate(["mesor_blup", "amplitude_blup", "acrophase_blup"]):
        for row, w in enumerate(WAVES):
            ax = axes[row, col]
            d = blups[w][param].to_numpy()
            d = d[~np.isnan(d)]
            ax.hist(d, bins=40, color="#4c78a8", edgecolor="white")
            ax.axvline(np.median(d), color="#d62728", lw=1.0, label=f"med {np.median(d):.2f}")
            ax.set_title(f"{w} — {param}", fontsize=10)
            ax.legend(frameon=False, fontsize=8)
            if param == "acrophase_blup":
                ax.set_xlabel("hours")
            elif param == "amplitude_blup":
                ax.set_xlabel("bpm")
            else:
                ax.set_xlabel("bpm")
    fig.tight_layout()
    fig.savefig(OUT / "fig_blup_distributions.png", dpi=120); plt.close(fig)


def fig_blup_pw_vs_pooled(blups_pw, blups_pooled):
    """Scatter per-wave BLUP vs pooled BLUP for each parameter × wave."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    pooled_pdf = blups_pooled.to_pandas()
    summary_corr = {}
    for col, param in enumerate(["mesor_blup", "amplitude_blup", "acrophase_blup"]):
        for row, w in enumerate(WAVES):
            ax = axes[row, col]
            pw = blups_pw[w].to_pandas()
            pl_w = pooled_pdf[pooled_pdf["wave"] == w]
            merged = pw.merge(pl_w, on="subject_id", suffixes=("_pw", "_pl"))
            x = merged[f"{param}_pw"].to_numpy()
            y = merged[f"{param}_pl"].to_numpy()
            ok = ~(np.isnan(x) | np.isnan(y))
            x, y = x[ok], y[ok]
            if param == "acrophase_blup":
                # Circular correlation: convert to radians, compute cos/sin corr
                ax_r = x / 24 * 2*np.pi; ay_r = y / 24 * 2*np.pi
                if x.size > 2:
                    a_bar = np.arctan2(np.sin(ax_r).mean(), np.cos(ax_r).mean())
                    b_bar = np.arctan2(np.sin(ay_r).mean(), np.cos(ay_r).mean())
                    num = np.sum(np.sin(ax_r - a_bar) * np.sin(ay_r - b_bar))
                    den = np.sqrt(np.sum(np.sin(ax_r - a_bar)**2) * np.sum(np.sin(ay_r - b_bar)**2))
                    rho = num / den if den > 0 else np.nan
                else:
                    rho = np.nan
                summary_corr[f"{w}/{param}"] = float(rho)
                lab = f"circ r={rho:.3f}"
            else:
                rho = float(np.corrcoef(x, y)[0,1]) if x.size > 1 else float("nan")
                summary_corr[f"{w}/{param}"] = rho
                lab = f"r={rho:.3f}"
            ax.scatter(x, y, s=2, alpha=0.25, color="#4c78a8")
            lo, hi = (np.nanmin(np.concatenate([x, y])) - 1, np.nanmax(np.concatenate([x, y])) + 1)
            ax.plot([lo, hi], [lo, hi], "--", color="#aaaaaa", lw=0.6)
            ax.set_xlabel("per-wave BLUP"); ax.set_ylabel("pooled BLUP")
            ax.set_title(f"{w} — {param}\n{lab}  (n={x.size:,})", fontsize=9)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    fig.tight_layout()
    fig.savefig(OUT / "fig_blup_per_wave_vs_pooled_scatter.png", dpi=120); plt.close(fig)
    return summary_corr


def fig_acrophase_polar(blups):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                              subplot_kw={"projection": "polar"})
    for j, w in enumerate(WAVES):
        ax = axes[j]
        d = blups[w]["acrophase_blup"].to_numpy()
        d = d[~np.isnan(d)]
        rad = (d / 24) * 2 * np.pi
        bins = np.linspace(0, 2*np.pi, 49)  # 30-min bins
        counts, edges = np.histogram(rad, bins=bins)
        widths = np.diff(edges)
        ax.bar((edges[:-1] + edges[1:]) / 2, counts, width=widths, color="#4c78a8", edgecolor="white")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)  # clockwise like a clock
        ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
        ax.set_title(f"{w}\nacrophase, N={d.size}", fontsize=9, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_acrophase_polar.png", dpi=120); plt.close(fig)


def fig_r2_distribution(blups):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for j, w in enumerate(WAVES):
        ax = axes[j]
        d = blups[w]["r_squared"].to_numpy()
        d = d[~np.isnan(d)]
        ax.hist(d, bins=40, color="#4c78a8", edgecolor="white")
        ax.axvline(0.10, color="#d62728", linestyle="--", lw=0.8, label="R² = 0.10")
        med = np.median(d); n_bad = int((d < 0.10).sum())
        ax.set_title(f"{w}\nN={d.size:,}, median R²={med:.3f}, R²<0.10: {n_bad}")
        ax.set_xlabel("per-participant R²")
        if j == 0: ax.set_ylabel("# subjects")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_r2_distribution.png", dpi=120); plt.close(fig)


def main() -> None:
    blups_pw, pops, diag = load_per_wave()
    blups_pl, pop_pooled, diag_pooled = load_pooled()

    fig_population_rhythm(blups_pw, pops)
    fig_blup_distributions(blups_pw)
    sens = fig_blup_pw_vs_pooled(blups_pw, blups_pl)
    fig_acrophase_polar(blups_pw)
    fig_r2_distribution(blups_pw)

    # Build chat summary
    summary = {"per_wave": {}, "sensitivity_correlations": sens, "pooled_diag": diag_pooled,
               "wave_to_wave_changes": {}}
    for w in WAVES:
        b = blups_pw[w]
        p = pops[w]
        d = diag[w]
        summary["per_wave"][w] = {
            "n_subjects": int(b.height),
            "n_poor_fit_r2_lt_0.10": int((b["r_squared"] < 0.10).sum()),
            "pop_mesor": p["fixed_effects"]["mesor"]["estimate"],
            "pop_mesor_ci95": p["fixed_effects"]["mesor"]["ci95"],
            "pop_amp": p["fixed_effects"]["amplitude"]["estimate"],
            "pop_amp_ci95": p["fixed_effects"]["amplitude"]["ci95"],
            "pop_acrophase_h": p["fixed_effects"]["acrophase"]["estimate_hours"],
            "convergence_code": d["convergence_code"],
            "formula_used": d["formula_used"],
            "runtime_seconds": d["runtime_seconds"],
            "blup_median": {
                "mesor": float(b["mesor_blup"].median()),
                "amplitude": float(b["amplitude_blup"].median()),
                "acrophase": float(b["acrophase_blup"].median()),
                "r_squared": float(b["r_squared"].median()),
            },
        }
    # Wave-to-wave changes (population)
    for k1, k2 in [("ses-02A","ses-04A"), ("ses-04A","ses-06A"), ("ses-02A","ses-06A")]:
        d_mesor = summary["per_wave"][k2]["pop_mesor"] - summary["per_wave"][k1]["pop_mesor"]
        d_amp   = summary["per_wave"][k2]["pop_amp"]   - summary["per_wave"][k1]["pop_amp"]
        d_acr   = summary["per_wave"][k2]["pop_acrophase_h"] - summary["per_wave"][k1]["pop_acrophase_h"]
        summary["wave_to_wave_changes"][f"{k1}->{k2}"] = {
            "delta_mesor": d_mesor, "delta_amplitude": d_amp, "delta_acrophase_hours": d_acr,
        }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
