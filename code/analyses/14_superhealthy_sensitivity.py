"""14 · Super-healthy HC sensitivity analysis.

Tests whether the primary single-outcome incidence findings hold when the
healthy-control reference group is restricted to "super-healthy"
participants: adolescents below the clinical threshold for ALL THREE primary
conditions (depression, obesity, hypertension) at every observed wave.
The primary analyses allow incident cases of one condition to serve as HCs
for another; this sensitivity removes that cross-condition contamination.

For each outcome (depression, obesity, hypertension):
    Cases (n unchanged)            = incident at W3 or W4 per the primary
                                       analytic frame
    Super-healthy HCs              = subjects in the primary analytic frame's
                                       HC group AND clean across all observed
                                       waves on all three conditions (i.e.,
                                       dep_lifetime == 0, obesity_lifetime ==
                                       0, htn_lifetime == 0; NaN flags
                                       excluded).

Predictors: three between-person cosinor parameters (mesor, amplitude,
acrophase) and three within-person SD features (SD daily mesor, amplitude,
acrophase). Each per-SD; logistic with age + sex + family-clustered SEs.
Multiple-comparison correction follows the primary convention (no BH-FDR);
p_fdr is recorded equal to p_raw.

Outputs:
    results/sensitivity/superhealthy_incidence.csv        (between-person)
    results/sensitivity/superhealthy_within_person.csv    (within-person)
    results/sensitivity/superhealthy_vs_primary_comparison.csv
    results/outputs/14_superhealthy_sensitivity.log
"""
from __future__ import annotations
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import (                              # noqa: E402
    TABLES_DIR, OUTPUTS_DIR, RESULTS_DIR, WITHIN_PERSON_FEATURES,
)
from utils.modeling import fit_logistic_cluster, fmt_or  # noqa: E402
from utils.cooccurrence import load_cooccurrence_frame  # noqa: E402

SENS_DIR = RESULTS_DIR / "sensitivity"
SENS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

FRAMES = [
    ("analytic_depression.tsv",   "Depression",   "primary_depression_results.tsv"),
    ("analytic_obesity.tsv",       "Obesity",       "primary_obesity_results.tsv"),
    ("analytic_hypertension.tsv",  "Hypertension",  "primary_hypertension_results.tsv"),
]
BETWEEN = [("typical_day_mesor",     "mesor"),
           ("typical_day_amplitude", "amplitude"),
           ("typical_day_acrophase", "acrophase")]
WITHIN = [("SD_daily_mesor",      "SD daily mesor"),
          ("SD_daily_amplitude",  "SD daily amplitude"),
          ("SD_daily_acrophase",  "SD daily acrophase")]
OR_SHIFT_THRESH = 0.05
P_FLIP_THRESH   = 0.05


def _discrepancy(or_a: float, or_b: float, p_a: float, p_b: float) -> str:
    if pd.isna(or_a) or pd.isna(or_b):
        return ""
    flags = []
    if abs(or_a - or_b) > OR_SHIFT_THRESH:
        flags.append(f"OR shift > {OR_SHIFT_THRESH}")
    sig_a = pd.notna(p_a) and p_a < P_FLIP_THRESH
    sig_b = pd.notna(p_b) and p_b < P_FLIP_THRESH
    if sig_a != sig_b:
        flags.append("sig flip at p < .05")
    return "; ".join(flags)


def main() -> None:
    out_lines: list[str] = []
    def log(msg: str = ""):
        print(msg); out_lines.append(msg)

    log("=" * 78)
    log("Super-healthy HC sensitivity")
    log("=" * 78)

    # ----- Section 1: define super-healthy HC pool -----
    co = load_cooccurrence_frame()
    sh_mask = ((co["dep_lifetime"] == 0)
                & (co["obesity_lifetime"] == 0)
                & (co["htn_lifetime"] == 0))
    sh_ids = set(co.loc[sh_mask, "participant_id"].tolist())
    log(f"\nSection 1. Super-healthy cohort definition")
    log(f"  Cosinor-pass cohort:                     n = {len(co):,}")
    log(f"  Super-healthy (clean on all 3, all obs): n = {len(sh_ids):,} "
        f"({100*len(sh_ids)/len(co):.1f}%)")

    # Within-person SD features (merge for section 2 within-person + section 4)
    wp_feat = (pd.read_csv(WITHIN_PERSON_FEATURES)
                  .rename(columns={"subject_id": "participant_id"})
                  [["participant_id", "SD_daily_mesor",
                    "SD_daily_amplitude", "SD_daily_acrophase"]])

    # ----- Section 2 + 4: re-run with super-healthy HCs -----
    log(f"\nSection 2 + 4. Logistic regression with super-healthy HCs")
    rows_btw: list[dict] = []
    rows_wp: list[dict] = []
    cell_summary: list[dict] = []

    for fname, outcome, _ in FRAMES:
        af = pd.read_csv(TABLES_DIR / fname, sep="\t").rename(columns={
            "mesor_blup":     "typical_day_mesor",
            "amplitude_blup": "typical_day_amplitude",
            "acrophase_blup": "typical_day_acrophase",
        })
        af = af.merge(wp_feat, on="participant_id", how="left")
        cases = af[af["onset"] == 1].copy()
        prim_hc = af[af["onset"] == 0].copy()
        super_hc = prim_hc[prim_hc["participant_id"].isin(sh_ids)].copy()
        sub = pd.concat([cases, super_hc], ignore_index=True)

        n_cases   = len(cases)
        n_hc_prim = len(prim_hc)
        n_hc_sh   = len(super_hc)
        pct_kept  = 100 * n_hc_sh / n_hc_prim if n_hc_prim > 0 else 0
        log(f"\n  === {outcome} ===")
        log(f"    n_cases = {n_cases};  n_HC primary = {n_hc_prim:,};  "
            f"n_HC super-healthy = {n_hc_sh:,} ({pct_kept:.1f}% kept)")

        cell_summary.append({"outcome": outcome, "n_cases": n_cases,
                               "n_hc_primary": n_hc_prim,
                               "n_hc_superhealthy": n_hc_sh,
                               "pct_kept": pct_kept})

        # Between-person
        for col, plabel in BETWEEN:
            r = fit_logistic_cluster(sub, [col], return_predictor=col)
            if r is None:
                log(f"    [between] {plabel:<12s} <skipped (n too small)>")
                continue
            log(f"    [between] {plabel:<12s} {fmt_or(r)}  "
                f"(events = {r.n_cases}, hc = {r.n - r.n_cases})")
            rows_btw.append({
                "outcome": outcome, "predictor": plabel,
                "or_per_sd": r.OR, "ci_lo": r.OR_lo, "ci_hi": r.OR_hi,
                "p_raw": r.p, "p_fdr": r.p,
                "n_cases": r.n_cases, "n_hc": r.n - r.n_cases,
            })

        # Within-person
        for col, plabel in WITHIN:
            sub_wp = sub.dropna(subset=[col])
            r = fit_logistic_cluster(sub_wp, [col], return_predictor=col)
            if r is None:
                log(f"    [within ] {plabel:<22s} <skipped (n too small)>")
                continue
            log(f"    [within ] {plabel:<22s} {fmt_or(r)}  "
                f"(events = {r.n_cases}, hc = {r.n - r.n_cases})")
            rows_wp.append({
                "outcome": outcome, "predictor": plabel,
                "or_per_sd": r.OR, "ci_lo": r.OR_lo, "ci_hi": r.OR_hi,
                "p_raw": r.p, "p_fdr": r.p,
                "n_cases": r.n_cases, "n_hc": r.n - r.n_cases,
            })

    btw_df = pd.DataFrame(rows_btw)
    wp_df  = pd.DataFrame(rows_wp)
    btw_df.to_csv(SENS_DIR / "superhealthy_incidence.csv", index=False)
    wp_df.to_csv(SENS_DIR / "superhealthy_within_person.csv", index=False)
    log(f"\n  Wrote {SENS_DIR / 'superhealthy_incidence.csv'}")
    log(f"  Wrote {SENS_DIR / 'superhealthy_within_person.csv'}")

    # ----- Section 3: side-by-side comparison vs primary -----
    log("\nSection 3. Side-by-side: primary vs super-healthy")
    cmp_rows: list[dict] = []
    PRED_KEY = {"mesor": "typical_day_mesor",
                  "amplitude": "typical_day_amplitude",
                  "acrophase": "typical_day_acrophase"}
    for fname, outcome, primary_file in FRAMES:
        primary = pd.read_csv(TABLES_DIR / primary_file, sep="\t")
        n_hc_prim = next(c["n_hc_primary"] for c in cell_summary
                           if c["outcome"] == outcome)
        n_hc_sh = next(c["n_hc_superhealthy"] for c in cell_summary
                         if c["outcome"] == outcome)
        log(f"\n  {outcome}:")
        for plabel, pcol in PRED_KEY.items():
            sh_row = btw_df[(btw_df["outcome"] == outcome)
                              & (btw_df["predictor"] == plabel)]
            if sh_row.empty:
                continue
            sh_row = sh_row.iloc[0]
            p_row = primary[(primary["analysis"] == "between")
                              & (primary["predictor"] == pcol)]
            if p_row.empty:
                continue
            p_row = p_row.iloc[0]
            flag = _discrepancy(p_row["OR"], sh_row["or_per_sd"],
                                  p_row["p"], sh_row["p_raw"])
            log(f"    {plabel:<10s}  "
                f"primary OR = {p_row['OR']:.2f} "
                f"[{p_row['OR_lo']:.2f}, {p_row['OR_hi']:.2f}], "
                f"p = {p_row['p']:.3g}  ||  "
                f"super-healthy OR = {sh_row['or_per_sd']:.2f} "
                f"[{sh_row['ci_lo']:.2f}, {sh_row['ci_hi']:.2f}], "
                f"p = {sh_row['p_raw']:.3g}"
                + (f"  <-- {flag}" if flag else ""))
            cmp_rows.append({
                "outcome": outcome, "predictor": plabel,
                "or_primary": p_row["OR"],
                "ci_primary_lo": p_row["OR_lo"], "ci_primary_hi": p_row["OR_hi"],
                "p_primary_fdr": p_row["p"],
                "or_superhealthy": sh_row["or_per_sd"],
                "ci_superhealthy_lo": sh_row["ci_lo"],
                "ci_superhealthy_hi": sh_row["ci_hi"],
                "p_superhealthy_fdr": sh_row["p_raw"],
                "n_hc_primary": n_hc_prim,
                "n_hc_superhealthy": n_hc_sh,
                "discrepancy_flag": flag,
            })
    pd.DataFrame(cmp_rows).to_csv(
        SENS_DIR / "superhealthy_vs_primary_comparison.csv", index=False)
    log(f"\n  Wrote {SENS_DIR / 'superhealthy_vs_primary_comparison.csv'}")

    (OUTPUTS_DIR / "14_superhealthy_sensitivity.log").write_text(
        "\n".join(out_lines))


if __name__ == "__main__":
    main()
