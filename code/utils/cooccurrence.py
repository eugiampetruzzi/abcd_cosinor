"""Co-occurrence cohort + lifetime-flag utilities.

Single source of truth for the four shared-signal scripts:
    08_cooccurrence_rates.py
    09_within_condition_stratification.py
    10_interaction_model.py
    11_conditional_prediction.py

Conventions:
    - Cosinor-pass cohort = subjects with W2 BLUPs in COSINOR_BLUP_W2.
    - Lifetime flag = 1 if subject ever above clinical threshold at any
      observed wave (W1-W4 for dep/obesity; W2-W4 for HTN), 0 if observed
      and never positive, NaN if never observed.
    - First-positive wave = earliest wave with the flag positive
      (NaN if never positive or never observed).

Renamed columns for downstream scripts:
    Between-person (typical-day BLUPs at W2):
        cosinor_mesor_w2, cosinor_amplitude_w2, cosinor_acrophase_w2
    Within-person (SD across daily cosinor fits, ≥7 valid wear days):
        sd_daily_mesor, sd_daily_amplitude, sd_daily_acrophase
    age_w2, is_female (0/1), family_id
    dep_lifetime, obesity_lifetime, htn_lifetime
    dep_first_wave, obesity_first_wave, htn_first_wave
    Per-W2 status (for conditional-prediction at-risk cohort definitions):
        dep_at_w2, obesity_at_w2, htn_at_w2  (Int64, NaN if not observed)
    Observed-after-W2 (any W3/W4 follow-up exists for the target):
        dep_obs_w3w4, obesity_obs_w3w4, htn_obs_w3w4   (bool)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from .paths import (
    COSINOR_BLUP_W2, MH_OUTCOMES, WITHIN_PERSON_FEATURES, W1, W2, W3, W4,
)
from .outcomes import load_mental_health, load_physical_health, load_sex, load_family


def _first_pos_wave(df_long: pd.DataFrame, flag: str,
                     waves: tuple[str, ...]) -> pd.Series:
    """Earliest wave (in `waves` order) where flag == 1; NaN otherwise."""
    out: dict[str, object] = {}
    for pid, g in df_long.dropna(subset=[flag]).groupby("participant_id"):
        first = pd.NA
        for w in waves:
            r = g[g["session_id"] == w]
            if r.empty:
                continue
            if int(r[flag].iloc[0]) == 1:
                first = w
                break
        out[pid] = first
    return pd.Series(out, name=f"{flag}_first_wave")


def _ever_pos(df_long: pd.DataFrame, flag: str,
                waves: tuple[str, ...]) -> pd.Series:
    """1 if ever positive; 0 if observed-and-negative-at-all-observed-waves;
    NaN if never observed at any wave in `waves`."""
    out: dict[str, object] = {}
    for pid, g in df_long.groupby("participant_id"):
        sub = g[g["session_id"].isin(waves)].dropna(subset=[flag])
        if sub.empty:
            out[pid] = pd.NA
        elif (sub[flag].astype(int) == 1).any():
            out[pid] = 1
        else:
            out[pid] = 0
    return pd.Series(out, name=f"ever_{flag}")


def _flag_at_wave(df_long: pd.DataFrame, flag: str, wave: str) -> pd.Series:
    """Per-subject 0/1/NaN for the flag at a single wave."""
    sub = df_long[df_long["session_id"] == wave][["participant_id", flag]]
    sub = sub.dropna(subset=[flag])
    sub[flag] = sub[flag].astype(int)
    return sub.set_index("participant_id")[flag]


def _observed_at_any(df_long: pd.DataFrame, flag: str,
                       waves: tuple[str, ...]) -> pd.Series:
    """Boolean: subject has at least one non-NaN observation of `flag` at one
    of the listed waves. Returns False for subjects with no rows in df_long."""
    sub = df_long[df_long["session_id"].isin(waves)].dropna(subset=[flag])
    return pd.Series(True, index=sub["participant_id"].unique(),
                       name=f"observed_{flag}")


def load_cooccurrence_frame() -> pd.DataFrame:
    """Return the canonical co-occurrence frame.

    Rows: cosinor-pass subjects (W2 BLUPs available).
    Columns:
        participant_id
        cosinor_mesor_w2, cosinor_amplitude_w2, cosinor_acrophase_w2
        age_w2, is_female, family_id
        dep_lifetime, obesity_lifetime, htn_lifetime          (Int64; NaN allowed)
        dep_first_wave, obesity_first_wave, htn_first_wave    (str; NaN allowed)
    """
    blup = (pl.read_parquet(COSINOR_BLUP_W2)
              .rename({"subject_id": "participant_id",
                        "mesor_blup": "cosinor_mesor_w2",
                        "amplitude_blup": "cosinor_amplitude_w2",
                        "acrophase_blup": "cosinor_acrophase_w2"})
              .select(["participant_id", "cosinor_mesor_w2",
                        "cosinor_amplitude_w2", "cosinor_acrophase_w2"])
              .to_pandas())

    sex = load_sex()
    fam = load_family()

    mh = load_mental_health()[["participant_id", "session_id",
                                 "cbcl_age", "dsm_dep_65"]].copy()
    age_w2 = (mh[mh["session_id"] == W2][["participant_id", "cbcl_age"]]
                .rename(columns={"cbcl_age": "age_w2"}))

    phys = load_physical_health(sex)[["participant_id", "session_id",
                                         "obese_85", "htn"]].copy()

    # Lifetime / first-positive: dep + obesity span W1-W4; HTN spans W2-W4.
    dep_first = _first_pos_wave(mh[["participant_id", "session_id",
                                       "dsm_dep_65"]], "dsm_dep_65",
                                  (W1, W2, W3, W4))
    dep_ever = _ever_pos(mh[["participant_id", "session_id", "dsm_dep_65"]],
                          "dsm_dep_65", (W1, W2, W3, W4))
    ob_first = _first_pos_wave(phys[["participant_id", "session_id",
                                        "obese_85"]], "obese_85",
                                 (W1, W2, W3, W4))
    ob_ever = _ever_pos(phys[["participant_id", "session_id", "obese_85"]],
                          "obese_85", (W1, W2, W3, W4))
    htn_first = _first_pos_wave(phys[["participant_id", "session_id",
                                         "htn"]], "htn",
                                  (W2, W3, W4))
    htn_ever = _ever_pos(phys[["participant_id", "session_id", "htn"]],
                           "htn", (W2, W3, W4))

    # Per-W2 status (0/1/NaN)
    dep_at_w2 = _flag_at_wave(mh[["participant_id", "session_id", "dsm_dep_65"]],
                                "dsm_dep_65", W2)
    ob_at_w2 = _flag_at_wave(phys[["participant_id", "session_id", "obese_85"]],
                                "obese_85", W2)
    htn_at_w2 = _flag_at_wave(phys[["participant_id", "session_id", "htn"]],
                                "htn", W2)
    # Observed-after-W2 (any W3 or W4 record present)
    dep_obs = _observed_at_any(mh[["participant_id", "session_id", "dsm_dep_65"]],
                                  "dsm_dep_65", (W3, W4))
    ob_obs = _observed_at_any(phys[["participant_id", "session_id", "obese_85"]],
                                 "obese_85", (W3, W4))
    htn_obs = _observed_at_any(phys[["participant_id", "session_id", "htn"]],
                                  "htn", (W3, W4))

    # Within-person SD features (subjects with ≥7 valid wear days)
    wp = (pd.read_csv(WITHIN_PERSON_FEATURES)
              .rename(columns={"subject_id": "participant_id"}))
    wp = wp[["participant_id",
              "SD_daily_mesor", "SD_daily_amplitude", "SD_daily_acrophase"]].rename(
        columns={"SD_daily_mesor":     "sd_daily_mesor",
                  "SD_daily_amplitude": "sd_daily_amplitude",
                  "SD_daily_acrophase": "sd_daily_acrophase"})

    df = (blup
            .merge(wp, on="participant_id", how="left")
            .merge(age_w2, on="participant_id", how="left")
            .merge(sex[["participant_id", "is_female"]],
                    on="participant_id", how="left")
            .merge(fam, on="participant_id", how="left")
            .merge(dep_ever.rename("dep_lifetime"),
                    left_on="participant_id", right_index=True, how="left")
            .merge(ob_ever.rename("obesity_lifetime"),
                    left_on="participant_id", right_index=True, how="left")
            .merge(htn_ever.rename("htn_lifetime"),
                    left_on="participant_id", right_index=True, how="left")
            .merge(dep_first.rename("dep_first_wave"),
                    left_on="participant_id", right_index=True, how="left")
            .merge(ob_first.rename("obesity_first_wave"),
                    left_on="participant_id", right_index=True, how="left")
            .merge(htn_first.rename("htn_first_wave"),
                    left_on="participant_id", right_index=True, how="left")
            .merge(dep_at_w2.rename("dep_at_w2"),
                    left_on="participant_id", right_index=True, how="left")
            .merge(ob_at_w2.rename("obesity_at_w2"),
                    left_on="participant_id", right_index=True, how="left")
            .merge(htn_at_w2.rename("htn_at_w2"),
                    left_on="participant_id", right_index=True, how="left")
            .merge(dep_obs.rename("dep_obs_w3w4"),
                    left_on="participant_id", right_index=True, how="left")
            .merge(ob_obs.rename("obesity_obs_w3w4"),
                    left_on="participant_id", right_index=True, how="left")
            .merge(htn_obs.rename("htn_obs_w3w4"),
                    left_on="participant_id", right_index=True, how="left"))
    # Cast lifetime + at_w2 flags to Int64 (nullable)
    for c in ["dep_lifetime", "obesity_lifetime", "htn_lifetime",
                "dep_at_w2", "obesity_at_w2", "htn_at_w2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    # observed-W3W4 flags: True if merged, False if NaN (no follow-up data)
    for c in ["dep_obs_w3w4", "obesity_obs_w3w4", "htn_obs_w3w4"]:
        df[c] = df[c].fillna(False).infer_objects(copy=False).astype(bool)
    return df
