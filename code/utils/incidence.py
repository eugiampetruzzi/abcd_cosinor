"""Canonical incident-diagnosis definition.

A participant is an INCIDENT CASE for a given outcome iff their first observed
clinically significant elevation occurred at Wave 3 (ses-04A) or Wave 4
(ses-06A), with documented absence of elevation at every earlier observed
assessment of that outcome:

    Wave-3 first onset:  clean at all `pre_waves` (when observed)
                        AND case at Wave 3
                        (Wave 4 status irrelevant; can be either)

    Wave-4 first onset:  clean at all `pre_waves` (when observed)
                        AND clean at Wave 3 (must be observed)
                        AND case at Wave 4

A participant is a CONTROL iff they have ≥1 observed Wave-3 or Wave-4
assessment AND were below the clinical threshold at every observed wave.

Wave-cleanness conventions per outcome:
    DSM-Depression  pre_waves = (W1=ses-00A, W2=ses-02A)
    Obesity         pre_waves = (W1=ses-00A, W2=ses-02A)
    Hypertension    pre_waves = (W2=ses-02A,)            # no BP at baseline

The `make_incidence_frame` function is the single source of truth: every
analysis (primary onset, within-person, sensitivity, etc.) imports it.
"""
from __future__ import annotations
from typing import Iterable

import pandas as pd

from .paths import W1, W2, W3, W4, PRE_WAVES_FULL, PRE_WAVES_BP


def make_incidence_frame(
    src: pd.DataFrame,
    flag: str,
    pre_waves: Iterable[str] = PRE_WAVES_FULL,
) -> pd.DataFrame:
    """Apply the canonical strict first-diagnosis rule.

    Parameters
    ----------
    src : long-format outcomes frame with columns
          {participant_id, session_id, <flag>}
    flag : str
          column name of the 0/1 elevation flag (e.g. 'dsm_dep_65')
    pre_waves : tuple[str, ...]
          waves where cleanness is required when observed (W2 always required)

    Returns
    -------
    DataFrame with columns:
        participant_id, onset (0/1), first_onset_wave (W3 / W4 / None for
        controls), definition ('strict')
    """
    pre_waves = tuple(pre_waves)
    if W2 not in pre_waves:
        raise ValueError(f"W2 ({W2}) must be included in pre_waves")

    sub = (src[src["session_id"].isin(list(pre_waves) + [W3, W4])]
              [["participant_id", "session_id", flag]]
              .dropna(subset=[flag]))
    rows: list[dict] = []
    for pid, g in sub.groupby("participant_id"):
        s = dict(zip(g["session_id"], g[flag]))

        # Must have a clean Wave-2 observation
        if W2 not in s or s[W2] == 1:
            continue
        # Cleanness at any other pre-wave that was observed
        if any(w in s and s[w] == 1 for w in pre_waves if w != W2):
            continue
        # Need ≥1 post-W2 follow-up
        if W3 not in s and W4 not in s:
            continue

        # Wave-3 first onset (W4 status irrelevant)
        if W3 in s and s[W3] == 1:
            rows.append({"participant_id": pid, "onset": 1,
                          "first_onset_wave": W3, "definition": "strict"})
            continue
        # Wave-4 first onset (must have observed AND clean Wave-3)
        if W4 in s and s[W4] == 1:
            if W3 in s and s[W3] == 0:
                rows.append({"participant_id": pid, "onset": 1,
                              "first_onset_wave": W4, "definition": "strict"})
            # If W3 missing, can't confirm first-onset → drop
            continue
        # Control: every observed post-W2 wave must be clean
        observed_post = [w for w in (W3, W4) if w in s]
        if observed_post and all(s[w] == 0 for w in observed_post):
            rows.append({"participant_id": pid, "onset": 0,
                          "first_onset_wave": None, "definition": "strict"})
    return pd.DataFrame(rows)


def incidence_summary(frame: pd.DataFrame) -> dict:
    """Counts to print / log."""
    n = len(frame)
    cs = frame[frame["onset"] == 1]
    n_w3 = int((cs["first_onset_wave"] == W3).sum())
    n_w4 = int((cs["first_onset_wave"] == W4).sum())
    return {
        "n_total": n,
        "n_cases": int(cs.shape[0]),
        "n_controls": n - int(cs.shape[0]),
        "n_W3_first_onsets": n_w3,
        "n_W4_first_onsets": n_w4,
    }
