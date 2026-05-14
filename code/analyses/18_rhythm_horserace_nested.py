"""18 · Rhythm horserace, nested models (super-healthy framework).

For each primary outcome (depression, obesity, hypertension), fit four nested
logistic models with family-clustered SEs, all on the same row set:

  M1 — HR cosinor only            (3 predictors)
  M2 — HR + activity cosinor      (9: + steps/METs × {mesor, amp, acro})
  M3 — HR + sleep cosinor         (6: + sleep × {mesor, amp, acro})
  M4 — HR + activity + sleep      (12)

All models add age + sex. Predictors z-scored within the outcome's analytic
sample. Super-healthy HCs (clean dep + obesity + HTN at every observed wave).

Reported:
  - OR [95% CI], p per predictor per model
  - log-lik / AIC / AUC per model
  - 5 LRTs: M1↔M2, M1↔M3, M1↔M4, M2↔M4, M3↔M4 (identical row set per outcome)
  - VIF per cosinor predictor per model
  - Pairwise Pearson r matrix of the 12 z-scored cosinor predictors per outcome

Sleep cosinor input: hourly proportion of wear-flagged minutes labeled asleep
(Fitbit Slp1m value ∈ {1, 2}), per-participant OLS on 24 hourly proportions.
Steps / METs cosinor: per-participant OLS on hourly mean per-minute streams.
HR cosinor: lme4 mixed-effects BLUPs from stage 3a.

Outputs:
    results/sensitivity/rhythm_horserace_nested.tsv          (predictor × model)
    results/sensitivity/rhythm_horserace_lrt.tsv             (LRT table)
    results/sensitivity/rhythm_horserace_vif.tsv             (VIF table)
    results/sensitivity/rhythm_horserace_corr_<slug>.tsv     (per-outcome corr)
    results/sensitivity/rhythm_horserace_summary.md
    figures/sensitivity/rhythm_horserace_full.{png,svg,pdf}
    figures/sensitivity/rhythm_horserace_hr_only.{png,svg,pdf}
    results/outputs/18_rhythm_horserace_nested.log
"""
from __future__ import annotations
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from scipy import stats as st
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import (                          # noqa: E402
    TABLES_DIR, OUTPUTS_DIR, RESULTS_DIR, FIGS_DIR, DERIV,
)

SENS_DIR = RESULTS_DIR / "sensitivity"
FIG_OUT  = FIGS_DIR / "sensitivity"
SENS_DIR.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

ACT_BLUPS = DERIV / "activity_cosinor" / "per_wave" / "ses-02A" / "participant_blups.parquet"

HR_COLS    = ["hr_mesor",    "hr_amplitude",    "hr_acrophase"]
STEPS_COLS = ["steps_mesor", "steps_amplitude", "steps_acrophase"]
METS_COLS  = ["mets_mesor",  "mets_amplitude",  "mets_acrophase"]
SLEEP_COLS = ["sleep_mesor", "sleep_amplitude", "sleep_acrophase"]
ALL_12     = HR_COLS + STEPS_COLS + METS_COLS + SLEEP_COLS

MODELS = {
    "M1_HR":          HR_COLS,
    "M2_HR+activity": HR_COLS + STEPS_COLS + METS_COLS,
    "M3_HR+sleep":    HR_COLS + SLEEP_COLS,
    "M4_full":        ALL_12,
}
LRT_PAIRS = [
    ("M1_HR", "M2_HR+activity"),
    ("M1_HR", "M3_HR+sleep"),
    ("M1_HR", "M4_full"),
    ("M2_HR+activity", "M4_full"),
    ("M3_HR+sleep",    "M4_full"),
]
FRAMES = [
    ("analytic_depression.tsv",   "Depression",   "dep"),
    ("analytic_obesity.tsv",      "Obesity",      "obesity"),
    ("analytic_hypertension.tsv", "Hypertension", "htn"),
]
LABEL = {
    "hr_mesor":        "HR mesor",
    "hr_amplitude":    "HR amplitude",
    "hr_acrophase":    "HR acrophase",
    "steps_mesor":     "Steps mesor",
    "steps_amplitude": "Steps amplitude",
    "steps_acrophase": "Steps acrophase",
    "mets_mesor":      "METs mesor",
    "mets_amplitude":  "METs amplitude",
    "mets_acrophase":  "METs acrophase",
    "sleep_mesor":     "Sleep mesor",
    "sleep_amplitude": "Sleep amplitude",
    "sleep_acrophase": "Sleep acrophase",
}


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def fit_model(df: pd.DataFrame, x_cols: list[str]) -> dict:
    """Logistic with family-clustered SEs; returns fit object + diagnostics."""
    use = df.dropna(subset=["onset", *x_cols,
                              "age_yrs", "is_female", "family_id"]).copy()
    use["onset"] = use["onset"].astype(int)
    Xz = use[x_cols + ["age_yrs", "is_female"]].copy()
    for c in x_cols:
        Xz[c] = _zscore(Xz[c])
    X = sm.add_constant(Xz, has_constant="add")
    f = sm.Logit(use["onset"], X).fit(
        disp=0, cov_type="cluster",
        cov_kwds={"groups": use["family_id"]}, maxiter=200)
    yhat = f.predict(X)
    return {
        "fit": f, "X": X, "use": use,
        "n":            int(f.nobs),
        "n_cases":      int(use["onset"].sum()),
        "loglik":       float(f.llf),
        "aic":          float(f.aic),
        "auc":          float(roc_auc_score(use["onset"], yhat)),
        "k_predictors": len(x_cols),
    }


def lrt(small: dict, big: dict) -> tuple[float, int, float]:
    if small["n"] != big["n"]:
        raise ValueError(f"LRT requires same row set: {small['n']} vs {big['n']}")
    stat = 2.0 * (big["loglik"] - small["loglik"])
    df   = big["k_predictors"] - small["k_predictors"]
    return stat, df, float(st.chi2.sf(stat, df))


def vif_table(X: pd.DataFrame) -> dict[str, float]:
    """VIF for every non-constant column in X."""
    arr = X.values.astype(float)
    return {c: float(variance_inflation_factor(arr, i))
              for i, c in enumerate(X.columns) if c != "const"}


def main() -> None:
    out_lines: list[str] = []
    def log(msg: str = ""):
        print(msg); out_lines.append(msg)

    log("=" * 78)
    log("Rhythm horserace, nested models (super-healthy framework)")
    log("=" * 78)

    act = (pl.read_parquet(ACT_BLUPS)
              .rename({"subject_id": "participant_id"})
              .to_pandas()[["participant_id",
                              *STEPS_COLS, *METS_COLS, *SLEEP_COLS]])
    log(f"  activity cosinor BLUPs available: n = {len(act):,} subjects")

    all_rows: list[dict] = []
    vif_rows: list[dict] = []
    lrt_rows: list[dict] = []
    summary: dict[str, dict] = {}

    for fname, label, slug in FRAMES:
        log(f"\n=== {label} ===")
        df = pd.read_csv(TABLES_DIR / fname, sep="\t")
        df = df.rename(columns={"mesor_blup":     "hr_mesor",
                                  "amplitude_blup": "hr_amplitude",
                                  "acrophase_blup": "hr_acrophase"})
        df = df.merge(act, on="participant_id", how="inner")
        df = df.dropna(subset=ALL_12 + ["age_yrs", "is_female",
                                          "family_id", "onset"]).copy()
        log(f"  shared row set across all 4 models: "
            f"n = {len(df):,}, cases = {int(df['onset'].sum())}")

        # Correlation matrix of the 12 z-scored cosinor predictors
        Z = df[ALL_12].apply(_zscore)
        corr = Z.corr().round(3)
        corr.index.name = "predictor"
        corr.to_csv(SENS_DIR / f"rhythm_horserace_corr_{slug}.tsv", sep="\t")
        off_diag_max = float(corr.abs().where(
            ~np.eye(12, dtype=bool)).max().max())
        log(f"  wrote rhythm_horserace_corr_{slug}.tsv "
            f"(|r|max off-diag = {off_diag_max:.3f})")

        fits: dict[str, dict] = {}
        for name, cols in MODELS.items():
            r = fit_model(df, cols)
            fits[name] = r
            log(f"  {name:<18s} k = {r['k_predictors']:>2d}  n = {r['n']:,}  "
                f"cases = {r['n_cases']}  log-lik = {r['loglik']:+.2f}  "
                f"AIC = {r['aic']:.1f}  AUC = {r['auc']:.3f}")
            # Per-predictor OR / CI / p
            for c in cols:
                b = float(r["fit"].params[c])
                ci = r["fit"].conf_int().loc[c].astype(float).tolist()
                all_rows.append({
                    "outcome": label, "model": name,
                    "predictor": c, "predictor_label": LABEL[c],
                    "n": r["n"], "n_cases": r["n_cases"],
                    "OR":    float(np.exp(b)),
                    "OR_lo": float(np.exp(ci[0])),
                    "OR_hi": float(np.exp(ci[1])),
                    "p":     float(r["fit"].pvalues[c]),
                })
            # VIF — restrict report to cosinor predictors
            vifs = vif_table(r["X"])
            for c, v in vifs.items():
                if c not in ALL_12:
                    continue
                vif_rows.append({
                    "outcome": label, "model": name,
                    "predictor": c, "predictor_label": LABEL[c],
                    "vif": v, "flag_5": v >= 5, "flag_10": v >= 10,
                })

        # LRTs
        log(f"  LRTs:")
        for a, b in LRT_PAIRS:
            stat, df_diff, p = lrt(fits[a], fits[b])
            d_auc = fits[b]["auc"] - fits[a]["auc"]
            d_aic = fits[b]["aic"] - fits[a]["aic"]
            log(f"    {a:>15s} → {b:<15s}  χ²({df_diff}) = {stat:.2f}, "
                f"p = {p:.3g}, ΔAUC = {d_auc:+.4f}, ΔAIC = {d_aic:+.1f}")
            lrt_rows.append({
                "outcome": label, "reduced": a, "full": b,
                "chi2": stat, "df": df_diff, "p": p,
                "auc_reduced": fits[a]["auc"],
                "auc_full":    fits[b]["auc"],
                "delta_auc": d_auc, "delta_aic": d_aic,
            })
        summary[label] = fits

    pd.DataFrame(all_rows).to_csv(
        SENS_DIR / "rhythm_horserace_nested.tsv", sep="\t", index=False)
    pd.DataFrame(lrt_rows).to_csv(
        SENS_DIR / "rhythm_horserace_lrt.tsv", sep="\t", index=False)
    pd.DataFrame(vif_rows).to_csv(
        SENS_DIR / "rhythm_horserace_vif.tsv", sep="\t", index=False)
    log(f"\nWrote {SENS_DIR / 'rhythm_horserace_nested.tsv'}")
    log(f"Wrote {SENS_DIR / 'rhythm_horserace_lrt.tsv'}")
    log(f"Wrote {SENS_DIR / 'rhythm_horserace_vif.tsv'}")

    # ---------------- Forest plots ----------------
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 10,
        "axes.linewidth": 0.9, "axes.edgecolor": "#222222",
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "savefig.dpi": 300, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    rows_df = pd.DataFrame(all_rows)

    # v1: all 12 ORs per outcome from M4 (busy but complete)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.6), sharey=True)
    y = np.arange(12)[::-1]
    for ax, (_, label, _) in zip(axes, FRAMES):
        sub = (rows_df[(rows_df["outcome"] == label)
                          & (rows_df["model"] == "M4_full")]
                  .set_index("predictor").loc[ALL_12])
        ax.axvline(1.0, color="#7A7A7A", lw=0.9, ls="--")
        ax.errorbar(sub["OR"], y,
                    xerr=[sub["OR"] - sub["OR_lo"], sub["OR_hi"] - sub["OR"]],
                    fmt="s", color="#E8635D", ecolor="#E8635D",
                    elinewidth=1.4, capsize=2.5, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[c] for c in ALL_12], fontsize=9)
        ax.set_xlabel("OR per SD")
        ax.set_title(f"{label}\n(n = {sub['n'].iloc[0]:,}; "
                     f"cases = {sub['n_cases'].iloc[0]})",
                     fontsize=10, color="#1F4E79")
    fig.suptitle("Model 4 — HR + activity + sleep cosinor (super-healthy HCs)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG_OUT / f"rhythm_horserace_full.{ext}",
                    bbox_inches="tight")
    plt.close(fig)
    log(f"Wrote {FIG_OUT / 'rhythm_horserace_full.png'}")

    # v2: HR predictors across the 4 nested models
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4), sharey=True)
    colors = {"M1_HR":          "#1F4E79",
              "M2_HR+activity": "#5E8FBF",
              "M3_HR+sleep":    "#8FB4D8",
              "M4_full":        "#E8635D"}
    y = np.arange(3)[::-1]
    for ax, (_, label, _) in zip(axes, FRAMES):
        ax.axvline(1.0, color="#7A7A7A", lw=0.9, ls="--")
        for j, m in enumerate(["M1_HR", "M2_HR+activity",
                                  "M3_HR+sleep", "M4_full"]):
            sub = (rows_df[(rows_df["outcome"] == label)
                              & (rows_df["model"] == m)
                              & (rows_df["predictor"].isin(HR_COLS))]
                      .set_index("predictor").loc[HR_COLS])
            offset = (j - 1.5) * 0.16
            ax.errorbar(sub["OR"], y + offset,
                        xerr=[sub["OR"] - sub["OR_lo"],
                                sub["OR_hi"] - sub["OR"]],
                        fmt="o", color=colors[m], ecolor=colors[m],
                        elinewidth=1.2, capsize=2.2, markersize=5.5,
                        markeredgecolor="white", markeredgewidth=0.6,
                        label=m if ax is axes[0] else None)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[c] for c in HR_COLS], fontsize=10)
        ax.set_xlabel("OR per SD")
        ax.set_title(label, fontsize=10, color="#1F4E79")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("HR rhythm OR across nested models", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG_OUT / f"rhythm_horserace_hr_only.{ext}",
                    bbox_inches="tight")
    plt.close(fig)
    log(f"Wrote {FIG_OUT / 'rhythm_horserace_hr_only.png'}")

    # ---------------- Markdown summary ----------------
    md: list[str] = ["# Rhythm horserace, nested models "
                      "(super-healthy framework)\n\n"]
    md.append(
        "All four models fit with family-clustered SEs, age + sex covariates, "
        "predictors z-scored within outcome's analytic sample. Super-healthy "
        "HCs (clean dep + obesity + HTN at every observed wave). All four "
        "models for an outcome share an identical row set (any participant "
        "missing any cosinor parameter dropped from all four).\n\n"
        "Sleep cosinor input: hourly proportion of wear-flagged minutes "
        "labeled asleep (Fitbit Slp1m value ∈ {1, 2}), per-participant OLS "
        "on 24 hourly proportions. Steps and METs cosinor: hourly mean per "
        "minute, OLS per participant. HR cosinor: lme4 mixed-effects BLUPs "
        "from stage 3a.\n\n"
    )

    md.append("## Model fits\n\n")
    md.append("| Outcome | Model | k | n | cases | log-lik | AIC | AUC |\n"
                "|---|---|---|---|---|---|---|---|\n")
    for label, fits in summary.items():
        for name, r in fits.items():
            md.append(f"| {label} | {name} | {r['k_predictors']} | "
                      f"{r['n']:,} | {r['n_cases']} | {r['loglik']:+.1f} | "
                      f"{r['aic']:.1f} | {r['auc']:.3f} |\n")

    md.append("\n## LRT comparisons (Δk = k_full − k_reduced)\n\n")
    md.append("| Outcome | Reduced | Full | Δk | χ² | p | ΔAUC | ΔAIC |\n"
                "|---|---|---|---|---|---|---|---|\n")
    for r in lrt_rows:
        md.append(f"| {r['outcome']} | {r['reduced']} | {r['full']} | "
                  f"{r['df']} | {r['chi2']:.2f} | {r['p']:.3g} | "
                  f"{r['delta_auc']:+.4f} | {r['delta_aic']:+.1f} |\n")

    md.append("\n## VIF — Model 4 (full 12-predictor model)\n\n")
    md.append("| Outcome | Predictor | VIF | Flag |\n|---|---|---|---|\n")
    for r in vif_rows:
        if r["model"] != "M4_full":
            continue
        flag = ("**VIF > 10**" if r["flag_10"]
                  else ("VIF > 5" if r["flag_5"] else ""))
        md.append(f"| {r['outcome']} | {r['predictor_label']} | "
                  f"{r['vif']:.2f} | {flag} |\n")

    md.append("\n## Pairwise |r| — top off-diagonal pairs per outcome\n\n")
    for fname, label, slug in FRAMES:
        c = pd.read_csv(SENS_DIR / f"rhythm_horserace_corr_{slug}.tsv",
                          sep="\t", index_col=0)
        pairs = []
        for i, ci in enumerate(c.columns):
            for j, cj in enumerate(c.columns):
                if j <= i: continue
                pairs.append((abs(c.iat[i, j]), c.iat[i, j], ci, cj))
        pairs.sort(reverse=True)
        md.append(f"### {label}\n\n")
        md.append("| Predictor A | Predictor B | r |\n|---|---|---|\n")
        for _, r_val, a, b in pairs[:8]:
            md.append(f"| {LABEL[a]} | {LABEL[b]} | {r_val:+.3f} |\n")
        md.append("\n")

    (SENS_DIR / "rhythm_horserace_summary.md").write_text("".join(md))
    log(f"Wrote {SENS_DIR / 'rhythm_horserace_summary.md'}")

    (OUTPUTS_DIR / "18_rhythm_horserace_nested.log").write_text(
        "\n".join(out_lines))


if __name__ == "__main__":
    main()
