"""01 · Sample refinement and incident-diagnosis definition.

Reproduces the participant flow and per-outcome incident-case counts in the
Methods text. Outputs:

    results/tables/sample_flow.tsv       — CONSORT-like attrition cascade
    results/tables/incident_cases.tsv    — per-outcome incidence counts
    results/outputs/01_sample_and_incidence.log

Run from the project root:
    python -m fitbit_prediction_github.code.analyses.01_sample_and_incidence
or directly:
    python fitbit_prediction_github/code/analyses/01_sample_and_incidence.py
"""
from __future__ import annotations
from pathlib import Path
import sys

import pandas as pd
import polars as pl

# Make `from utils...` importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import (                          # noqa: E402
    TABLES_DIR, OUTPUTS_DIR, COSINOR_BLUP_W2,
    W1, W2, W3, W4, PRE_WAVES_FULL, PRE_WAVES_BP,
)
from utils.outcomes import (                       # noqa: E402
    load_mental_health, load_physical_health, load_sex, load_family,
)
from utils.cohort import build_cohort              # noqa: E402
from utils.incidence import (                       # noqa: E402
    make_incidence_frame, incidence_summary,
)

TABLES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def assemble_analytic(
    incidence_df: pd.DataFrame,
    blups: pd.DataFrame,
    age_w2: pd.DataFrame,
    sex_pd: pd.DataFrame,
    family: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join incidence frame with the covariates needed for modelling.

    Drops rows missing any of: cosinor BLUP, age at W2, sex, family_id.
    """
    df = (incidence_df.merge(blups, on="participant_id", how="inner")
                       .merge(age_w2, on="participant_id", how="left")
                       .merge(sex_pd[["participant_id", "is_female"]],
                               on="participant_id", how="left")
                       .merge(family, on="participant_id", how="left"))
    df = df.dropna(subset=["mesor_blup", "age_yrs", "is_female", "family_id"])
    return df


def main() -> None:
    out_lines: list[str] = []
    def log(msg: str = ""):
        print(msg); out_lines.append(msg)

    log("=" * 78)
    log("Step 1–3: Participant-flow cascade")
    log("=" * 78)

    sex     = load_sex()
    family  = load_family()
    mh      = load_mental_health()
    phys    = load_physical_health(sex=sex)

    cohort = build_cohort(mh, phys)
    flow_rows = [
        ("Wave-2 Novel Technologies sub-study enrolled",
         len(cohort["enrolled_w2"])),
        ("Met Fitbit data-quality criteria (4-of-4, ≥3 days)",
         len(cohort["wearable_pass"])),
        ("Has Wave-2 cosinor BLUP (sanity check)",
         len(cohort["wearable_blup"])),
        ("Has any outcome at W3 or W4 (CBCL or BMI or BP)",
         len(cohort["any_outcome_w3_or_w4"])),
        ("Analytic pool (wearable-pass ∩ has-outcome)",
         len(cohort["analytic_pool"])),
    ]
    for lbl, n in flow_rows:
        log(f"  {lbl:<55s}  n = {n:>5,}")

    flow_tbl = pd.DataFrame(flow_rows, columns=["step", "n"])
    flow_tbl.to_csv(TABLES_DIR / "sample_flow.tsv", sep="\t", index=False)
    log(f"\nWrote {TABLES_DIR / 'sample_flow.tsv'}")

    # ----- Per-outcome strict incidence -----
    log("\n" + "=" * 78)
    log("Per-outcome strict incidence (canonical definition)")
    log("=" * 78)

    blups = (pl.read_parquet(COSINOR_BLUP_W2)
                .filter(pl.col("r_squared").is_not_null())
                .select(["subject_id", "mesor_blup",
                          "amplitude_blup", "acrophase_blup"])
                .rename({"subject_id": "participant_id"})
                .to_pandas())

    OUTCOMES = [
        # (label, flag, src, pre_waves, slug, age_col_at_W2)
        ("DSM Depression", "dsm_dep_65", mh,   PRE_WAVES_FULL, "depression",   "cbcl_age"),
        ("Obesity",        "obese_85",   phys, PRE_WAVES_FULL, "obesity",      "anthr_age"),
        ("Hypertension",   "htn",        phys, PRE_WAVES_BP,   "hypertension", "bp_age"),
    ]

    incidence_rows: list[dict] = []
    for label, flag, src, pre_waves, slug, age_col in OUTCOMES:
        age_w2 = (src[src["session_id"] == W2][["participant_id", age_col]]
                    .rename(columns={age_col: "age_yrs"})
                    .drop_duplicates("participant_id"))
        log(f"\n--- {label} ---")
        inc = make_incidence_frame(src, flag, pre_waves)
        sm = incidence_summary(inc)
        log(f"  Strict incidence frame:        "
            f"n = {sm['n_total']:>5,}   cases = {sm['n_cases']:>5}   "
            f"(W3 first: {sm['n_W3_first_onsets']}, "
            f"W4 first: {sm['n_W4_first_onsets']})")

        analytic = assemble_analytic(inc, blups, age_w2, sex, family)
        n_a = len(analytic)
        n_c = int(analytic["onset"].sum())
        n_w3 = int((analytic["first_onset_wave"] == W3).sum())
        n_w4 = int((analytic["first_onset_wave"] == W4).sum())
        log(f"  Analytic sample (∩ covariates): "
            f"n = {n_a:>5,}   cases = {n_c:>5}   "
            f"(W3 first: {n_w3}, W4 first: {n_w4})")

        # Save the analytic frame for downstream analyses
        analytic.to_csv(TABLES_DIR / f"analytic_{slug}.tsv", sep="\t", index=False)
        log(f"  wrote analytic_{slug}.tsv")

        incidence_rows.append({
            "outcome": label,
            "pre_waves_required_clean": ",".join(pre_waves),
            "strict_n_total": sm["n_total"],
            "strict_n_cases": sm["n_cases"],
            "strict_W3_first": sm["n_W3_first_onsets"],
            "strict_W4_first": sm["n_W4_first_onsets"],
            "analytic_n": n_a,
            "analytic_cases": n_c,
            "analytic_W3_first": n_w3,
            "analytic_W4_first": n_w4,
        })

    inc_tbl = pd.DataFrame(incidence_rows)
    inc_tbl.to_csv(TABLES_DIR / "incident_cases.tsv", sep="\t", index=False)
    log(f"\nWrote {TABLES_DIR / 'incident_cases.tsv'}")

    # ----- Manuscript-text verification -----
    log("\n" + "=" * 78)
    log("Verification against Methods text")
    log("=" * 78)
    expected = {
        "Wave-2 sub-study enrolled":     8_166,
        "Met wearable quality":          7_230,
        "Has any outcome at W3/W4":      6_601,
        "Depression analytic n":         4_205,
        "Depression analytic cases":       463,
        "Obesity analytic n":            3_932,
        "Obesity analytic cases":          457,
        "Hypertension analytic n":       2_669,
        "Hypertension analytic cases":     270,
    }
    actual = {
        "Wave-2 sub-study enrolled":     len(cohort["enrolled_w2"]),
        "Met wearable quality":          len(cohort["wearable_pass"]),
        "Has any outcome at W3/W4":      len(cohort["analytic_pool"]),
        "Depression analytic n":         incidence_rows[0]["analytic_n"],
        "Depression analytic cases":     incidence_rows[0]["analytic_cases"],
        "Obesity analytic n":            incidence_rows[1]["analytic_n"],
        "Obesity analytic cases":        incidence_rows[1]["analytic_cases"],
        "Hypertension analytic n":       incidence_rows[2]["analytic_n"],
        "Hypertension analytic cases":   incidence_rows[2]["analytic_cases"],
    }
    log(f"  {'Step':<30s}  {'expected':>9s}  {'actual':>9s}  {'match':>5s}")
    for k in expected:
        e, a = expected[k], actual[k]
        log(f"  {k:<30s}  {e:>9,}  {a:>9,}  {'  OK' if e == a else 'MISMATCH'}")

    (OUTPUTS_DIR / "01_sample_and_incidence.log").write_text("\n".join(out_lines))
    log(f"\nWrote {OUTPUTS_DIR / '01_sample_and_incidence.log'}")


if __name__ == "__main__":
    main()
