"""15 · Demographic comparison: super-healthy HCs vs. standard-but-not-
super-healthy HCs.

Characterizes whether super-healthy HCs (clean across all three primary
conditions at every observed wave) differ from standard HCs who would NOT
qualify as super-healthy (HC in at least one primary analytic frame but
incident on at least one of the other conditions during follow-up). Useful
for reviewer-anticipated generalizability concerns about the super-healthy
sensitivity analysis (Script 14).

Source for demographics: a pre-extracted file already curated by Eu that
collapses ab_p_demo.tsv + ab_g_stc.tsv:
    /Users/.../Release 6.1/Actigraphy_Eu_Outputs/subject_demographics.tsv

Columns used (verified present in that file):
    sex_label, race_label, ethn_label, ethnrace_label,
    income_3lvl_label, edu_cgs_label, site_baseline, age_02A

Additional variables not in that file:
    pubertal stage (pds_max_categ at Wave 2)   from load_physical_health()
    Wave-2 Fitbit covariates                  from per_wave_summary.parquet:
        sleep_period_min, waso_min, daily_steps, mets_avg

Variables explicitly NOT included (flagged for review):
    Household composition (single- vs two-parent) — not in the pre-extracted
      file; can be added later from ab_p_demo if needed.
    Family structure flags (sibling/twin/triplet) — not pulled here.
    Age at Wave 1 — only 145/9343 of the Fitbit cohort were in the W1 pilot;
      using Wave-2 age (age_02A) as the de-facto baseline.

For continuous variables: Welch's t-test + Cohen's d.
For categorical variables: chi-square + Cramer's V.
Bonferroni applied across all variables tested.

Outputs:
    results/sensitivity/superhealthy_demographics.csv
    results/outputs/15_superhealthy_demographics.log
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import (                              # noqa: E402
    TABLES_DIR, OUTPUTS_DIR, RESULTS_DIR, DERIV, ONEDRIVE, W2,
)
from utils.cooccurrence import load_cooccurrence_frame  # noqa: E402
from utils.outcomes import load_physical_health, load_sex  # noqa: E402

SENS_DIR = RESULTS_DIR / "sensitivity"
SENS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEMO_TSV = (ONEDRIVE / "Release 6.1" / "Actigraphy_Eu_Outputs"
                / "subject_demographics.tsv")
PER_WAVE_SUMMARY = DERIV / "fitbit_summary" / "per_wave_summary.parquet"

CONT_VARS = [
    ("age_02A",          "Age at Wave 2 (yr)"),
    ("pds_max_categ",    "Pubertal stage (PDS, Wave 2)"),
    ("sleep_period_min", "Wave-2 mean daily sleep (min)"),
    ("waso_min",         "Wave-2 mean daily WASO (min)"),
    ("daily_steps",      "Wave-2 mean daily steps"),
    ("mets_avg",         "Wave-2 mean daily METs"),
]
CAT_VARS = [
    ("sex_label",         "Sex"),
    ("ethnrace_label",     "Race/ethnicity (collapsed)"),
    ("income_3lvl_label",  "Household income (3 levels)"),
    ("edu_cgs_label",      "Parent education"),
    ("site_baseline",      "Study site"),
]


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna(); b = b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                  / (na + nb - 2))
    if s == 0:
        return np.nan
    return float((a.mean() - b.mean()) / s)


def interp_d(d: float) -> str:
    a = abs(d)
    if a < 0.20: return "negligible"
    if a < 0.50: return "small"
    if a < 0.80: return "medium"
    return "large"


def cramers_v(tbl: np.ndarray) -> float:
    chi2 = st.chi2_contingency(tbl)[0]
    n = tbl.sum()
    r, c = tbl.shape
    return float(np.sqrt(chi2 / (n * min(r - 1, c - 1)))) if n > 0 else np.nan


def interp_v(v: float) -> str:
    if v < 0.10: return "negligible"
    if v < 0.30: return "small"
    if v < 0.50: return "medium"
    return "large"


def compare_continuous(df: pd.DataFrame, var: str, group_col: str
                        ) -> dict:
    a = df.loc[df[group_col] == "super_healthy", var].dropna()
    b = df.loc[df[group_col] == "not_super_healthy_HC", var].dropna()
    if len(a) < 2 or len(b) < 2:
        return None
    t, p = st.ttest_ind(a, b, equal_var=False)
    d = cohen_d(a, b)
    return {
        "variable_type": "continuous",
        "super_healthy_summary": f"{a.mean():.2f} (SD {a.std():.2f})",
        "super_healthy_n": int(len(a)),
        "not_super_healthy_summary": f"{b.mean():.2f} (SD {b.std():.2f})",
        "not_super_healthy_n": int(len(b)),
        "test": "Welch t",
        "statistic": float(t),
        "p_value": float(p),
        "effect_size_name": "Cohen's d",
        "effect_size": d,
        "effect_size_interp": interp_d(d) if d == d else "",
    }


def compare_categorical(df: pd.DataFrame, var: str, group_col: str
                          ) -> dict:
    sub = df[[var, group_col]].dropna()
    tbl = pd.crosstab(sub[var], sub[group_col])
    if "super_healthy" not in tbl.columns or "not_super_healthy_HC" not in tbl.columns:
        return None
    tbl = tbl[["super_healthy", "not_super_healthy_HC"]]
    a = tbl["super_healthy"]; b = tbl["not_super_healthy_HC"]
    na = int(a.sum()); nb = int(b.sum())
    chi2, p, dof, _ = st.chi2_contingency(tbl.values)
    v = cramers_v(tbl.values)
    # Distribution string (top 5 categories shown)
    def _dist(s, total):
        return "; ".join(
            f"{k}: {int(v)} ({100*v/total:.0f}%)"
            for k, v in s.sort_values(ascending=False).head(5).items())
    return {
        "variable_type": "categorical",
        "super_healthy_summary": _dist(a, na),
        "super_healthy_n": na,
        "not_super_healthy_summary": _dist(b, nb),
        "not_super_healthy_n": nb,
        "test": f"chi² ({dof})",
        "statistic": float(chi2),
        "p_value": float(p),
        "effect_size_name": "Cramer's V",
        "effect_size": v,
        "effect_size_interp": interp_v(v),
    }


def main() -> None:
    out_lines: list[str] = []
    def log(msg: str = ""):
        print(msg); out_lines.append(msg)

    log("=" * 78)
    log("Super-healthy vs. standard-but-not-super-healthy HC demographics")
    log("=" * 78)

    # ----- Build the two groups -----
    co = load_cooccurrence_frame()
    sh_mask = ((co["dep_lifetime"] == 0)
                & (co["obesity_lifetime"] == 0)
                & (co["htn_lifetime"] == 0))
    super_ids = set(co.loc[sh_mask, "participant_id"].tolist())

    dep = pd.read_csv(TABLES_DIR / "analytic_depression.tsv", sep="\t")
    ob  = pd.read_csv(TABLES_DIR / "analytic_obesity.tsv",     sep="\t")
    htn = pd.read_csv(TABLES_DIR / "analytic_hypertension.tsv", sep="\t")
    hc_anywhere = (set(dep.loc[dep["onset"] == 0, "participant_id"])
                    | set(ob.loc[ob["onset"]   == 0, "participant_id"])
                    | set(htn.loc[htn["onset"] == 0, "participant_id"]))
    co_pos_any = co[
        (co["dep_lifetime"] == 1)
        | (co["obesity_lifetime"] == 1)
        | (co["htn_lifetime"] == 1)
    ]["participant_id"].tolist()
    not_sh_hc_ids = (set(co_pos_any) & hc_anywhere) - super_ids

    log(f"\n  Super-healthy HCs:                  n = {len(super_ids):,}")
    log(f"  Standard-but-not-super-healthy HCs: n = {len(not_sh_hc_ids):,}")
    log(f"    (Of {len(co):,} cosinor-pass total; "
        f"{len(hc_anywhere):,} are HCs in at least one primary frame.)")

    # ----- Assemble per-subject demographics frame -----
    demo = pd.read_csv(DEMO_TSV, sep="\t")
    sex = load_sex()
    phys = load_physical_health(sex)
    pds = (phys[phys["session_id"] == W2][["participant_id", "pds_max_categ"]]
               .copy())
    pds["pds_max_categ"] = pd.to_numeric(pds["pds_max_categ"], errors="coerce")
    cov = (pl.read_parquet(PER_WAVE_SUMMARY)
              .filter(pl.col("session_id") == "ses-02A")
              .select(["participant_id", "sleep_period_min", "waso_min",
                        "daily_steps", "mets_avg"])
              .to_pandas())

    df = (pd.DataFrame({"participant_id": list(super_ids | not_sh_hc_ids)})
            .merge(demo, on="participant_id", how="left")
            .merge(pds, on="participant_id", how="left")
            .merge(cov, on="participant_id", how="left"))
    df["group"] = df["participant_id"].map(
        lambda pid: "super_healthy" if pid in super_ids
                    else "not_super_healthy_HC")

    # ----- Run comparisons -----
    rows: list[dict] = []
    log("\nContinuous comparisons:")
    for var, label in CONT_VARS:
        if var not in df.columns:
            log(f"  {label:<35s}  [variable missing]")
            continue
        r = compare_continuous(df, var, "group")
        if r is None:
            log(f"  {label:<35s}  [insufficient n]")
            continue
        rows.append({"variable": label, **r})
        log(f"  {label:<35s}  super-healthy: {r['super_healthy_summary']} "
            f"(n={r['super_healthy_n']:,})  |  "
            f"not-SH HC: {r['not_super_healthy_summary']} "
            f"(n={r['not_super_healthy_n']:,})  |  "
            f"t = {r['statistic']:+.2f}, p = {r['p_value']:.3g}, "
            f"d = {r['effect_size']:+.3f} ({r['effect_size_interp']})")

    log("\nCategorical comparisons:")
    for var, label in CAT_VARS:
        if var not in df.columns:
            log(f"  {label:<35s}  [variable missing]")
            continue
        r = compare_categorical(df, var, "group")
        if r is None:
            log(f"  {label:<35s}  [insufficient n]")
            continue
        rows.append({"variable": label, **r})
        log(f"  {label:<35s}  n_SH = {r['super_healthy_n']:,}, "
            f"n_notSH = {r['not_super_healthy_n']:,};  "
            f"chi² = {r['statistic']:.2f}, p = {r['p_value']:.3g}, "
            f"V = {r['effect_size']:.3f} ({r['effect_size_interp']})")
        log(f"    super-healthy top categories:  {r['super_healthy_summary']}")
        log(f"    not-super-healthy top:         {r['not_super_healthy_summary']}")

    out_df = pd.DataFrame(rows)
    # Bonferroni across all tested variables
    n_tests = len(out_df)
    out_df["p_bonferroni"] = (out_df["p_value"] * n_tests).clip(upper=1.0)
    out_df.to_csv(SENS_DIR / "superhealthy_demographics.csv", index=False)
    log(f"\nBonferroni multiplier: {n_tests}")
    log(f"Wrote {SENS_DIR / 'superhealthy_demographics.csv'}")

    # ----- Effect-size flag summary -----
    log("\nEffect-size flags (continuous |d| ≥ 0.20 or categorical V ≥ 0.10):")
    flagged = out_df[(
        ((out_df["variable_type"] == "continuous")
            & (out_df["effect_size"].abs() >= 0.20))
        | ((out_df["variable_type"] == "categorical")
            & (out_df["effect_size"] >= 0.10))
    )]
    if flagged.empty:
        log("  (none above threshold)")
    else:
        for _, r in flagged.iterrows():
            log(f"  {r['variable']:<35s}  "
                f"{r['effect_size_name']} = {r['effect_size']:+.3f} "
                f"({r['effect_size_interp']}); "
                f"p = {r['p_value']:.3g}; p_bonf = {r['p_bonferroni']:.3g}")

    (OUTPUTS_DIR / "15_superhealthy_demographics.log").write_text(
        "\n".join(out_lines))


if __name__ == "__main__":
    main()
