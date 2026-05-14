"""Sample-refinement cascade.

Implements the participant flow described in the Methods text:

    1. ABCD Wave-2 Novel Technologies sub-study enrollment      (n = 8,166)
       -> all participants with any ses-02A Fitbit session in the cohort
          inventory (`qc/stage1_full_cohort_sessions.tsv`)
    2. Met wearable-data quality criteria                        (n = 7,230)
       -> session passes_4of4_min3days at ses-02A
       -> equivalently: has a non-null Wave-2 cosinor BLUP
    3. Has any outcome data at Wave 3 or Wave 4                 (n = 6,601)
       -> CBCL DSM-Depression observed at ses-04A or ses-06A,
          OR BMI observed at ses-04A or ses-06A,
          OR blood pressure observed at ses-04A or ses-06A.

After step 3, per-outcome incident-case selection (see `incidence.py`)
applies. The intersection with covariates required by the regression
(family_id, sex, age) is left to each analysis script.
"""
from __future__ import annotations

import pandas as pd
import polars as pl

from .paths import (
    QC, COSINOR_BLUP_W2,
    W2, W3, W4,
)


# ---------------------------------------------------------------------------
# Step 1 + 2 — Wearable cohort
# ---------------------------------------------------------------------------

def load_w2_substudy_enrolled() -> set[str]:
    """All participants with ANY Wave-2 Fitbit session in the cohort manifest.

    Source: qc/stage1_full_cohort_sessions.tsv (one row per session).
    """
    sess = pl.read_csv(QC / "stage1_full_cohort_sessions.tsv", separator="\t")
    return set(sess.filter(pl.col("session_id") == W2)["subject_id"].to_list())


def load_wearable_quality_pass() -> set[str]:
    """Participants whose Wave-2 session met wearable-quality criteria.

    Definition: passes_4of4_min3days at ses-02A — i.e. ≥3 valid wear days
    where each day has HR coverage in all four 6-h quadrants.
    """
    sess = pl.read_csv(QC / "stage1_full_cohort_sessions.tsv", separator="\t")
    pri = sess.filter((pl.col("session_id") == W2)
                       & pl.col("passes_4of4_min3days"))
    return set(pri["subject_id"].to_list())


def load_w2_blup_subjects() -> set[str]:
    """Participants with a non-null Wave-2 cosinor BLUP (sanity-check that
    matches `load_wearable_quality_pass`)."""
    blups = (pl.read_parquet(COSINOR_BLUP_W2)
                .filter(pl.col("r_squared").is_not_null())
                .select("subject_id"))
    return set(blups["subject_id"].to_list())


# ---------------------------------------------------------------------------
# Step 3 — Has any outcome data at W3 or W4
# ---------------------------------------------------------------------------

def has_any_outcome_w3_or_w4(mh: pd.DataFrame, phys: pd.DataFrame) -> set[str]:
    """Participants with ≥1 observed outcome (CBCL Dep, BMI, or BP) at W3/W4."""
    has = set()
    mh_post = mh[mh["session_id"].isin([W3, W4])]
    has |= set(mh_post.dropna(subset=["cbcl_dsm_dep_tscore"])["participant_id"])
    phys_post = phys[phys["session_id"].isin([W3, W4])]
    has |= set(phys_post.dropna(subset=["bmi"])["participant_id"])
    has |= set(phys_post.dropna(subset=["bp_sys_mean", "bp_dia_mean"])["participant_id"])
    return has


# ---------------------------------------------------------------------------
# Convenience: full cascade returning a dict of counts + sets
# ---------------------------------------------------------------------------

def build_cohort(mh: pd.DataFrame, phys: pd.DataFrame) -> dict:
    """Return the participant-flow cascade as counts and sets.

    Keys:
        enrolled_w2          — all W2 sub-study participants            (~8,166)
        wearable_pass        — met Fitbit data-quality criteria         (~7,230)
        wearable_blup        — has a Wave-2 cosinor BLUP (sanity)
        any_outcome_w3_or_w4 — has CBCL/BMI/BP at W3 or W4              (~6,601)
        analytic_pool        — wearable_pass ∩ any_outcome_w3_or_w4
    """
    enrolled  = load_w2_substudy_enrolled()
    pass_qc   = load_wearable_quality_pass()
    has_blup  = load_w2_blup_subjects()
    has_outc  = has_any_outcome_w3_or_w4(mh, phys)
    return {
        "enrolled_w2":          enrolled,
        "wearable_pass":        pass_qc,
        "wearable_blup":        has_blup,
        "any_outcome_w3_or_w4": has_outc,
        "analytic_pool":        pass_qc & has_outc,
    }
