"""16 · HC vs. incident-case demographic comparison.

For the Sample Characteristics paragraph of the manuscript, compare the
super-healthy HC group against each of the three incident-case groups
(depression, obesity, hypertension) on age, sex, race/ethnicity, and
household income. Format and variable set match the preceding sentence
that compared the analytic cohort to ABCD non-analytic participants.

Groups (drawn from the cosinor-pass cohort, N = 7,230):
    HC (super-healthy)     — below threshold for dep + obesity + HTN at
                              every observed wave (N ≈ 2,987)
    Depression cases       — first elevation of CBCL DSM-Depression at W3 or W4
                              (N = 463)
    Obesity cases          — first elevation of BMI ≥ 85th pct at W3 or W4
                              (N = 457)
    Hypertension cases     — first elevation of BP per CDC simplified criteria
                              at W3 or W4 (N = 270)
Incident groups may overlap; each participant is included in every incident
group they qualify for.

For each HC vs. incident comparison:
    age (continuous)       — Welch's t-test + Cohen's d
    sex / race / income    — chi-square + Cramer's V

Outputs:
    results/sample_characteristics/hc_vs_incident_demographics.csv
    results/outputs/16_hc_vs_incident_demographics.log
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import TABLES_DIR, OUTPUTS_DIR, RESULTS_DIR, ONEDRIVE  # noqa: E402
from utils.cooccurrence import load_cooccurrence_frame  # noqa: E402

OUT_DIR = RESULTS_DIR / "sample_characteristics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DEMO_TSV = (ONEDRIVE / "Release 6.1" / "Actigraphy_Eu_Outputs"
                / "subject_demographics.tsv")


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna(); b = b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                  / (na + nb - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def cramers_v(tbl: np.ndarray) -> float:
    chi2 = st.chi2_contingency(tbl)[0]
    n = tbl.sum(); r, c = tbl.shape
    return float(np.sqrt(chi2 / (n * min(r - 1, c - 1)))) if n > 0 else np.nan


def interp_d(d: float) -> str:
    a = abs(d) if d == d else 0
    if a < 0.20: return "negligible"
    if a < 0.50: return "small"
    if a < 0.80: return "medium"
    return "large"


def interp_v(v: float) -> str:
    if v != v: return ""
    if v < 0.10: return "negligible"
    if v < 0.30: return "small"
    if v < 0.50: return "medium"
    return "large"


def compare_continuous(hc: pd.DataFrame, inc: pd.DataFrame, var: str
                          ) -> dict:
    a = hc[var].dropna(); b = inc[var].dropna()
    t, p = st.ttest_ind(a, b, equal_var=False)
    d = cohen_d(a, b)
    return {
        "variable_type": "continuous",
        "hc_value": f"{a.mean():.2f} (SD {a.std():.2f})",
        "incident_value": f"{b.mean():.2f} (SD {b.std():.2f})",
        "hc_n": len(a), "incident_n": len(b),
        "test_statistic": float(t),
        "test_name": "Welch t",
        "p_value": float(p),
        "effect_size": d,
        "effect_size_type": "Cohen's d",
        "effect_size_interp": interp_d(d),
        "flag_small_effect": (abs(d) >= 0.20) if d == d else False,
    }


def compare_categorical(hc: pd.DataFrame, inc: pd.DataFrame, var: str,
                          highlight_value: str = None) -> dict:
    a = hc[var].dropna(); b = inc[var].dropna()
    combined = np.concatenate([a.values, b.values])
    group = np.array(["hc"] * len(a) + ["incident"] * len(b))
    tbl = pd.crosstab(combined, group)
    chi2, p, dof, _ = st.chi2_contingency(tbl.values)
    v = cramers_v(tbl.values)
    # n (%) per category, focus on highlight if specified
    def _pct(s, val):
        return 100 * float((s == val).sum()) / len(s) if len(s) > 0 else np.nan
    summary_lines_hc, summary_lines_inc = [], []
    for cat in tbl.index:
        summary_lines_hc.append(
            f"{cat}: {int((a == cat).sum())} ({100*float((a == cat).sum())/len(a):.1f}%)")
        summary_lines_inc.append(
            f"{cat}: {int((b == cat).sum())} ({100*float((b == cat).sum())/len(b):.1f}%)")
    out = {
        "variable_type": "categorical",
        "hc_value": "; ".join(summary_lines_hc),
        "incident_value": "; ".join(summary_lines_inc),
        "hc_n": len(a), "incident_n": len(b),
        "test_statistic": float(chi2),
        "test_name": f"chi² ({dof})",
        "p_value": float(p),
        "effect_size": v,
        "effect_size_type": "Cramer's V",
        "effect_size_interp": interp_v(v),
        "flag_small_effect": (v >= 0.10) if v == v else False,
    }
    if highlight_value:
        out["hc_highlight_pct"] = _pct(a, highlight_value)
        out["incident_highlight_pct"] = _pct(b, highlight_value)
        out["highlight_category"] = highlight_value
    return out


def main() -> None:
    out_lines: list[str] = []
    def log(msg: str = ""):
        print(msg); out_lines.append(msg)

    log("=" * 78)
    log("HC vs. incident-case demographic comparisons")
    log("=" * 78)

    # ----- Groups (new common-HC framework) -----
    # HCs differ across outcomes: the depression/obesity HC pool is identical
    # (clean on all three at every observed wave + has CBCL & BMI at W3/W4);
    # the hypertension HC pool is a subset that also has BP at W3/W4.
    hc_ids = {}
    case_ids = {}
    for slug, label in [("depression", "depression"),
                          ("obesity", "obesity"),
                          ("hypertension", "hypertension")]:
        af = pd.read_csv(TABLES_DIR / f"analytic_{slug}.tsv", sep="\t")
        hc_ids[label]   = set(af.loc[af["onset"] == 0, "participant_id"].tolist())
        case_ids[label] = set(af.loc[af["onset"] == 1, "participant_id"].tolist())
        log(f"  Outcome {label}: HC n = {len(hc_ids[label]):,}, "
            f"incident cases n = {len(case_ids[label]):,}")
    # Overlap check
    dep_ob = case_ids["depression"] & case_ids["obesity"]
    dep_htn = case_ids["depression"] & case_ids["hypertension"]
    ob_htn = case_ids["obesity"] & case_ids["hypertension"]
    all3 = case_ids["depression"] & case_ids["obesity"] & case_ids["hypertension"]
    log(f"  Overlap dep ∩ obesity: {len(dep_ob):,}; "
        f"dep ∩ HTN: {len(dep_htn):,}; "
        f"obesity ∩ HTN: {len(ob_htn):,}; "
        f"all three: {len(all3):,}")

    # ----- Demographics -----
    demo = pd.read_csv(DEMO_TSV, sep="\t")

    # Map race_label to standard category set matching preceding sentence
    # (White, Black, Hispanic, Asian, Other/Multiracial). The pre-extracted
    # ethnrace_label collapses race × ethnicity. Use race_label and treat
    # NaN as missing.
    cont_vars = [("age_02A", "Age at Wave 2 (yr)")]
    cat_vars = [
        ("sex_label",         "Sex",                "Female"),
        ("ethnrace_label",     "Race/ethnicity",     "White"),
        ("income_3lvl_label", "Annual household income", ">100k"),
    ]

    rows: list[dict] = []
    summary_per_group: dict[str, list[str]] = {}

    for case_label in ["depression", "obesity", "hypertension"]:
        log(f"\n--- HC vs. incident {case_label} ---")
        hc_demo  = demo[demo["participant_id"].isin(hc_ids[case_label])].copy()
        inc_demo = demo[demo["participant_id"].isin(
            case_ids[case_label])].copy()
        log(f"  HC n = {len(hc_demo):,};  incident-{case_label} n = "
            f"{len(inc_demo):,}")
        summary_per_group[case_label] = []
        for var, label in cont_vars:
            r = compare_continuous(hc_demo, inc_demo, var)
            rows.append({"comparison": f"HC vs. {case_label}",
                          "variable": label, **r})
            log(f"  {label:<28s}  HC = {r['hc_value']}  |  "
                f"incident = {r['incident_value']}  |  "
                f"t = {r['test_statistic']:+.2f}, p = {r['p_value']:.3g}, "
                f"d = {r['effect_size']:+.3f} ({r['effect_size_interp']})"
                + ("  *" if r["flag_small_effect"] else ""))
            if r["flag_small_effect"]:
                summary_per_group[case_label].append(
                    f"age (HC {hc_demo[var].mean():.1f}, "
                    f"incident {inc_demo[var].mean():.1f} yr; d = {r['effect_size']:+.2f})")

        for var, label, hl in cat_vars:
            r = compare_categorical(hc_demo, inc_demo, var, highlight_value=hl)
            rows.append({"comparison": f"HC vs. {case_label}",
                          "variable": label, **r})
            log(f"  {label:<28s}  HC: {r['hc_value']}")
            log(f"  {' ':<28s}  incident: {r['incident_value']}")
            log(f"  {' ':<28s}  chi² = {r['test_statistic']:.2f}, "
                f"p = {r['p_value']:.3g}, V = {r['effect_size']:.3f} "
                f"({r['effect_size_interp']})"
                + ("  *" if r["flag_small_effect"] else ""))
            if r["flag_small_effect"]:
                summary_per_group[case_label].append(
                    f"{label.lower()} ({hl}: "
                    f"HC {r.get('hc_highlight_pct', np.nan):.1f}%, "
                    f"incident {r.get('incident_highlight_pct', np.nan):.1f}%; "
                    f"V = {r['effect_size']:.2f})")

    # ----- Save -----
    pd.DataFrame(rows).to_csv(
        OUT_DIR / "hc_vs_incident_demographics.csv", index=False)
    log(f"\nWrote {OUT_DIR / 'hc_vs_incident_demographics.csv'}")

    # ----- Manuscript-ready sentence -----
    log("\n" + "=" * 78)
    log("Manuscript-ready sentence (paste into Sample Characteristics):")
    log("=" * 78)

    def _seg(g: str, n_inc: int) -> str:
        feats = summary_per_group[g]
        if not feats:
            return f"did not differ meaningfully from incident {g} cases (N = {n_inc})"
        return (f"differed from incident {g} cases (N = {n_inc}) on "
                + " and ".join(feats))

    n_dep_ob_hc = len(hc_ids["depression"])  # same as hc_ids["obesity"]
    n_htn_hc    = len(hc_ids["hypertension"])
    s = (f"Healthy controls (N = {n_dep_ob_hc:,} for depression and obesity; "
         f"N = {n_htn_hc:,} for hypertension) "
          + _seg('depression', len(case_ids['depression'])) + "; "
          + _seg('obesity', len(case_ids['obesity'])) + "; and "
          + _seg('hypertension', len(case_ids['hypertension'])) + ".")
    log(s)

    (OUTPUTS_DIR / "16_hc_vs_incident_demographics.log").write_text(
        "\n".join(out_lines))


if __name__ == "__main__":
    main()
