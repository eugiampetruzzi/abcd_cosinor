"""18b · Stability of HTN M4 HR amplitude coefficient.

The hypertension M4 HR amplitude OR shifts from 0.81 (M1) → 0.56 (M4) — a
large suppression effect. With n=1,459 and 263 cases / 14 predictors,
EPV ≈ 19 (tight). This script verifies the M4 amplitude coefficient is not
driven by a small subset of observations.

Diagnostics:
  1. Family-clustered bootstrap (B=1000): refit M4 on resampled families,
     extract HR amplitude OR; report 2.5/97.5 percentiles vs the analytic CI.
  2. One-step dfBetas: per-observation influence on the HR amplitude
     coefficient. Re-fit M4 dropping top 5, 10, 20 influential observations
     and report the resulting OR.
  3. Leave-one-site-out: refit M4 with each ABCD site excluded; report
     the spread of HR amplitude OR.

Outputs:
  results/sensitivity/htn_amplitude_stability.md
  results/outputs/18b_htn_amplitude_stability.log
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import (                          # noqa: E402
    TABLES_DIR, OUTPUTS_DIR, RESULTS_DIR, DERIV, ONEDRIVE,
)

SENS_DIR = RESULTS_DIR / "sensitivity"
SENS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

ACT_BLUPS = DERIV / "activity_cosinor" / "per_wave" / "ses-02A" / "participant_blups.parquet"
DEMO_TSV  = ONEDRIVE / "Release 6.1" / "Actigraphy_Eu_Outputs" / "subject_demographics.tsv"

HR_COLS    = ["hr_mesor", "hr_amplitude", "hr_acrophase"]
STEPS_COLS = ["steps_mesor", "steps_amplitude", "steps_acrophase"]
METS_COLS  = ["mets_mesor", "mets_amplitude", "mets_acrophase"]
SLEEP_COLS = ["sleep_mesor", "sleep_amplitude", "sleep_acrophase"]
ALL_12     = HR_COLS + STEPS_COLS + METS_COLS + SLEEP_COLS

B_BOOT = 1000
SEED   = 20260505


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def build_design(df: pd.DataFrame, x_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    use = df.dropna(subset=["onset", *x_cols,
                              "age_yrs", "is_female", "family_id"]).copy()
    use["onset"] = use["onset"].astype(int)
    Xz = use[x_cols + ["age_yrs", "is_female"]].copy()
    for c in x_cols:
        Xz[c] = _zscore(Xz[c])
    X = sm.add_constant(Xz, has_constant="add")
    X.index = use.index
    return X, use["onset"]


def fit_m4(use_df: pd.DataFrame, x_cols: list[str], cluster: pd.Series | None = None):
    """Return statsmodels Logit fit on the given frame (z-scored within)."""
    X, y = build_design(use_df, x_cols)
    kwargs = dict(disp=0, maxiter=200)
    if cluster is not None:
        kwargs.update(cov_type="cluster", cov_kwds={"groups": cluster.loc[X.index]})
    return sm.Logit(y, X).fit(**kwargs), X, y


def main() -> None:
    out_lines: list[str] = []
    def log(msg: str = ""):
        print(msg); out_lines.append(msg)

    log("=" * 78)
    log("HTN M4 HR amplitude stability diagnostics")
    log("=" * 78)

    # Build the same row set used in script 18
    act = (pl.read_parquet(ACT_BLUPS)
              .rename({"subject_id": "participant_id"})
              .to_pandas()[["participant_id",
                              *STEPS_COLS, *METS_COLS, *SLEEP_COLS]])
    df = pd.read_csv(TABLES_DIR / "analytic_hypertension.tsv", sep="\t")
    df = df.rename(columns={"mesor_blup":     "hr_mesor",
                              "amplitude_blup": "hr_amplitude",
                              "acrophase_blup": "hr_acrophase"})
    df = df.merge(act, on="participant_id", how="inner")
    df = df.dropna(subset=ALL_12 + ["age_yrs", "is_female",
                                      "family_id", "onset"]).copy()
    df = df.reset_index(drop=True)
    log(f"  HTN M4 analytic n = {len(df):,}, cases = {int(df['onset'].sum())}")

    # ------ Analytic baseline (clustered SEs) ------
    fit, X, y = fit_m4(df, ALL_12, cluster=df["family_id"])
    b_analytic = float(fit.params["hr_amplitude"])
    ci_analytic = fit.conf_int().loc["hr_amplitude"].astype(float).tolist()
    or_analytic = float(np.exp(b_analytic))
    log(f"\n  Analytic M4 HR amplitude:  OR = {or_analytic:.3f} "
        f"[{np.exp(ci_analytic[0]):.3f}, {np.exp(ci_analytic[1]):.3f}], "
        f"p = {fit.pvalues['hr_amplitude']:.3g}")

    # ------ 1. Family-clustered bootstrap ------
    log("\n--- 1. Family-clustered bootstrap (B = "
        f"{B_BOOT}, resample families with replacement) ---")
    rng = np.random.default_rng(SEED)
    fam_groups = df.groupby("family_id").indices  # family_id → np.array of row indices
    fam_ids = list(fam_groups.keys())
    n_fam = len(fam_ids)
    log(f"    n unique families = {n_fam:,}")

    boots: list[float] = []
    fail = 0
    for b in range(B_BOOT):
        chosen = rng.choice(n_fam, size=n_fam, replace=True)
        rows = np.concatenate([fam_groups[fam_ids[i]] for i in chosen])
        sub = df.iloc[rows].copy()
        if sub["onset"].nunique() < 2:
            fail += 1; continue
        try:
            f, _, _ = fit_m4(sub, ALL_12, cluster=None)
            boots.append(float(np.exp(f.params["hr_amplitude"])))
        except Exception:
            fail += 1
        if (b + 1) % 200 == 0:
            log(f"    bootstrap {b+1:>4d}/{B_BOOT}  "
                f"(median so far = {np.median(boots):.3f})")

    arr = np.asarray(boots)
    log(f"    successful refits: {len(arr)}/{B_BOOT}  (failures: {fail})")
    log(f"    bootstrap OR median = {np.median(arr):.3f}")
    log(f"    bootstrap OR mean   = {arr.mean():.3f}  (SE = {arr.std():.3f})")
    log(f"    bootstrap 95% CI    = [{np.percentile(arr, 2.5):.3f}, "
        f"{np.percentile(arr, 97.5):.3f}]")
    log(f"    bootstrap 99% CI    = [{np.percentile(arr, 0.5):.3f}, "
        f"{np.percentile(arr, 99.5):.3f}]")
    log(f"    % bootstraps with OR < 1: {100*(arr < 1).mean():.1f}%")
    log(f"    % bootstraps with OR < 0.7: {100*(arr < 0.7).mean():.1f}%")

    # ------ 2. dfBetas / influence ------
    log("\n--- 2. Per-observation influence (one-step dfBetas) ---")
    # Use unclustered fit for influence (closed-form influence requires
    # standard SEs; we want which observations matter most for the point
    # estimate, not the SE).
    f0, X0, y0 = fit_m4(df, ALL_12, cluster=None)
    or_unclust = float(np.exp(f0.params["hr_amplitude"]))
    log(f"    Unclustered M4 HR amplitude OR (reference): {or_unclust:.3f}")
    # One-step dfBetas via the influence module (logit-aware)
    infl = f0.get_influence()
    dfb = pd.DataFrame(infl.dfbetas, columns=X0.columns,
                        index=X0.index)
    amp_dfb = dfb["hr_amplitude"].abs().sort_values(ascending=False)
    log(f"    Top 10 |dfBetas| on HR amplitude:")
    for idx in amp_dfb.head(10).index:
        log(f"      row {int(idx):>5d}  pid={df.loc[idx, 'participant_id']}  "
            f"onset={int(df.loc[idx, 'onset'])}  "
            f"hr_amp_z = {(df.loc[idx, 'hr_amplitude'] - df['hr_amplitude'].mean())/df['hr_amplitude'].std():+.2f}  "
            f"|dfBeta| = {amp_dfb[idx]:.4f}")

    # Refit dropping top N influential
    for n_drop in (5, 10, 20):
        keep = ~df.index.isin(amp_dfb.head(n_drop).index)
        sub = df[keep]
        f_d, _, _ = fit_m4(sub, ALL_12, cluster=sub["family_id"])
        or_d = float(np.exp(f_d.params["hr_amplitude"]))
        ci_d = f_d.conf_int().loc["hr_amplitude"].astype(float).tolist()
        log(f"    Drop top {n_drop:>2d} influential:  n = {len(sub):,}, "
            f"OR = {or_d:.3f} [{np.exp(ci_d[0]):.3f}, "
            f"{np.exp(ci_d[1]):.3f}], p = {f_d.pvalues['hr_amplitude']:.3g}")

    # ------ 3. Leave-one-site-out ------
    log("\n--- 3. Leave-one-site-out ---")
    demo = pd.read_csv(DEMO_TSV, sep="\t",
                         usecols=["participant_id", "site_baseline"])
    df_s = df.merge(demo, on="participant_id", how="left")
    sites = sorted(df_s["site_baseline"].dropna().unique().tolist())
    log(f"    n sites = {len(sites)}; participants with site missing: "
        f"{df_s['site_baseline'].isna().sum()}")
    log(f"    {'Excluded':<14s} {'n':>6s} {'cases':>6s} "
        f"{'OR':>7s} {'95% CI':>20s} {'p':>10s}")
    los_ors = []
    for site in sites:
        site_s = str(site)
        sub = df_s[df_s["site_baseline"] != site]
        if sub["onset"].nunique() < 2 or sub["onset"].sum() < 10:
            log(f"    {site_s:<14s}  too few cases, skip")
            continue
        f_s, _, _ = fit_m4(sub, ALL_12, cluster=sub["family_id"])
        or_s = float(np.exp(f_s.params["hr_amplitude"]))
        ci_s = f_s.conf_int().loc["hr_amplitude"].astype(float).tolist()
        los_ors.append(or_s)
        log(f"    {site_s:<14s} {len(sub):>6,d} {int(sub['onset'].sum()):>6d} "
            f"{or_s:>7.3f} [{np.exp(ci_s[0]):.3f}, {np.exp(ci_s[1]):.3f}] "
            f"{f_s.pvalues['hr_amplitude']:>10.3g}")
    if los_ors:
        log(f"\n    Leave-one-site-out OR range: "
            f"[{min(los_ors):.3f}, {max(los_ors):.3f}]; "
            f"median = {np.median(los_ors):.3f}")

    # ------ Markdown summary ------
    md = [
        "# HTN M4 HR amplitude — stability diagnostics\n\n",
        f"Analytic M4 HR amplitude (super-healthy framework, family-clustered SEs): "
        f"**OR = {or_analytic:.3f} [{np.exp(ci_analytic[0]):.3f}, "
        f"{np.exp(ci_analytic[1]):.3f}], p = {fit.pvalues['hr_amplitude']:.3g}** "
        f"(n = {len(df):,}, cases = {int(df['onset'].sum())}).\n\n",
        "## Family-clustered bootstrap\n\n",
        f"- {len(arr):,} successful refits of {B_BOOT} (families resampled with replacement)\n",
        f"- Bootstrap median OR = **{np.median(arr):.3f}**\n",
        f"- 95% percentile CI = **[{np.percentile(arr, 2.5):.3f}, "
        f"{np.percentile(arr, 97.5):.3f}]**\n",
        f"- % bootstraps with OR < 1: {100*(arr < 1).mean():.1f}%; "
        f"% with OR < 0.7: {100*(arr < 0.7).mean():.1f}%\n\n",
        "## Influence (one-step dfBetas)\n\n",
        f"Reference unclustered OR = {or_unclust:.3f}.\n\n",
        "| Drop top N | n | OR | 95% CI | p |\n|---|---|---|---|---|\n",
    ]
    for n_drop in (5, 10, 20):
        keep = ~df.index.isin(amp_dfb.head(n_drop).index)
        sub = df[keep]
        f_d, _, _ = fit_m4(sub, ALL_12, cluster=sub["family_id"])
        or_d = float(np.exp(f_d.params["hr_amplitude"]))
        ci_d = f_d.conf_int().loc["hr_amplitude"].astype(float).tolist()
        md.append(f"| {n_drop} | {len(sub):,} | {or_d:.3f} | "
                  f"[{np.exp(ci_d[0]):.3f}, {np.exp(ci_d[1]):.3f}] | "
                  f"{f_d.pvalues['hr_amplitude']:.3g} |\n")
    md.append("\n## Leave-one-site-out\n\n")
    if los_ors:
        md.append(f"OR range across {len(los_ors)} sites: "
                  f"**[{min(los_ors):.3f}, {max(los_ors):.3f}]**, "
                  f"median = {np.median(los_ors):.3f}.\n")
    (SENS_DIR / "htn_amplitude_stability.md").write_text("".join(md))
    log(f"\nWrote {SENS_DIR / 'htn_amplitude_stability.md'}")
    (OUTPUTS_DIR / "18b_htn_amplitude_stability.log").write_text(
        "\n".join(out_lines))


if __name__ == "__main__":
    main()
