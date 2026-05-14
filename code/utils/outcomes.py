"""Outcome-flag construction.

Loads the master longitudinal frames and computes binary clinical-elevation
flags at each wave for the three primary outcomes:

    dsm_dep_65 : CBCL DSM-Depression T-score >= 65            (mental health)
    obese_85   : CDC BMI percentile >= 85th (CDC LMS table)   (anthropometric)
    htn        : SBP >= 130 OR DBP >= 80 mmHg                 (blood pressure)

Each loader returns a long-format `participant_id × session_id × flag` table
(dropping rows with no measurement). Demographics + family structure are also
provided as a join table for downstream modelling.

Convention: every consumer downstream filters / pivots from these long tables;
no other module re-derives the flag columns.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats as st

from .paths import (
    MH_OUTCOMES, PHYS_OUTCOMES, DEMO_DIR, FAMILY_PARQUET, CDC_BMI_LMS,
)


# ---------------------------------------------------------------------------
# Demographics + family
# ---------------------------------------------------------------------------

def load_sex() -> pd.DataFrame:
    """Returns participant_id × {is_female (0/1), sex_cdc (1=M, 2=F)}."""
    stc = pl.read_csv(
        DEMO_DIR / "ab_g_stc.tsv", separator="\t",
        null_values=["n/a", ""], infer_schema_length=10000,
        ignore_errors=True,
    ).to_pandas()
    return pd.DataFrame({
        "participant_id": stc["participant_id"],
        "is_female": stc["ab_g_stc__cohort_sex"].apply(
            lambda v: 1 if v == 2 else (0 if v == 1 else np.nan)),
        "sex_cdc": stc["ab_g_stc__cohort_sex"].apply(
            lambda v: int(v) if v in (1, 2) else np.nan),
    })


def load_family() -> pd.DataFrame:
    """Returns participant_id × family_id (Int64), for cluster-robust SEs."""
    return (pl.read_parquet(FAMILY_PARQUET)
              .with_columns(pl.col("family_id").cast(pl.Int64))
              .to_pandas())


# ---------------------------------------------------------------------------
# Mental health (CBCL DSM-Depression)
# ---------------------------------------------------------------------------

def load_mental_health() -> pd.DataFrame:
    """Returns long-format MH frame with `dsm_dep_65` 0/1 flag.

    Output columns: participant_id, session_id, cbcl_dsm_dep_tscore,
                    cbcl_age, dsm_dep_65
    """
    mh = pl.read_parquet(MH_OUTCOMES).to_pandas()
    mh["dsm_dep_65"] = mh["cbcl_dsm_dep_tscore"].apply(
        lambda v: pd.NA if pd.isna(v) else (1 if v >= 65 else 0))
    return mh


# ---------------------------------------------------------------------------
# Physical health (BMI -> obese_85, BP -> htn)
# ---------------------------------------------------------------------------

def _bmi_pct_fn():
    """Return a vectorisable closure that computes CDC BMI percentile."""
    lms = pd.read_csv(CDC_BMI_LMS); lms = lms[lms["Sex"] != "Sex"].copy()
    for c in lms.columns:
        lms[c] = pd.to_numeric(lms[c], errors="coerce")

    def bmi_pct(bmi: float, age_months: float, sex_cdc: int) -> float:
        if (pd.isna(bmi) or pd.isna(age_months) or pd.isna(sex_cdc)
                or age_months < 24 or age_months > 240):
            return np.nan
        sub = lms[lms["Sex"] == sex_cdc]
        idx = (sub["Agemos"] - age_months).abs().idxmin()
        L, M, S = sub.at[idx, "L"], sub.at[idx, "M"], sub.at[idx, "S"]
        if abs(L) < 1e-6:
            z = np.log(bmi / M) / S
        else:
            z = ((bmi / M) ** L - 1.0) / (L * S)
        return float(st.norm.cdf(z) * 100)
    return bmi_pct


def load_physical_health(sex: pd.DataFrame | None = None) -> pd.DataFrame:
    """Returns long-format physical-health frame with `obese_85` and `htn` flags.

    Requires `sex` (with `sex_cdc`) to compute CDC BMI percentile. If not
    supplied, calls `load_sex()` itself.

    Output columns: participant_id, session_id,
                    bmi, bmi_pct, anthr_age, obese_85,
                    bp_sys_mean, bp_dia_mean, bp_age, htn
    """
    phys = pl.read_parquet(PHYS_OUTCOMES).to_pandas()
    if sex is None:
        sex = load_sex()
    phys = phys.merge(sex[["participant_id", "sex_cdc"]],
                       on="participant_id", how="left")
    phys["age_months"] = phys["anthr_age"] * 12
    bmi_pct = _bmi_pct_fn()
    phys["bmi_pct"] = phys.apply(
        lambda r: bmi_pct(r["bmi"], r["age_months"], r["sex_cdc"]), axis=1)
    phys["obese_85"] = (phys["bmi_pct"] >= 85).astype("Int64")
    phys.loc[phys["bmi_pct"].isna(), "obese_85"] = pd.NA
    phys["htn"] = ((phys["bp_sys_mean"] >= 130)
                    | (phys["bp_dia_mean"] >= 80)).astype("Int64")
    phys.loc[phys["bp_sys_mean"].isna()
              | phys["bp_dia_mean"].isna(), "htn"] = pd.NA
    return phys
