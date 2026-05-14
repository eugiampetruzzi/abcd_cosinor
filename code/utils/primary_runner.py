"""Shared primary-analysis runner used by `03_primary_depression.py`,
`04_primary_obesity.py`, and `05_primary_hypertension.py`.

Encapsulates the modelling framework described in the Methods:

    Logistic regression on z-scored predictor + age + sex with cluster-robust
    SEs on family_id. Run for each between-person cosinor BLUP and each
    within-person stability feature, then jointly with the typical-day mesor
    to test incremental signal.

Returns a dict that the calling script can use for both the TSV output and
inline reporting in the manuscript text.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from .modeling import fit_logistic_cluster, fmt_or
from .paths import TABLES_DIR, OUTPUTS_DIR, WITHIN_PERSON_FEATURES


BETWEEN = [
    ("typical_day_mesor",     "Typical-day mesor"),
    ("typical_day_amplitude", "Typical-day amplitude"),
    ("typical_day_acrophase", "Typical-day acrophase"),
]
WITHIN = [
    ("SD_daily_mesor",     "SD daily mesor"),
    ("SD_daily_amplitude", "SD daily amplitude"),
    ("SD_daily_acrophase", "SD daily acrophase"),
]


def load_analytic_frame(slug: str) -> pd.DataFrame:
    """Load the per-outcome analytic frame produced by 01_… and merge in the
    within-person SD features."""
    df = pd.read_csv(TABLES_DIR / f"analytic_{slug}.tsv", sep="\t")
    df = df.rename(columns={"mesor_blup": "typical_day_mesor",
                              "amplitude_blup": "typical_day_amplitude",
                              "acrophase_blup": "typical_day_acrophase"})
    feats = (pd.read_csv(WITHIN_PERSON_FEATURES)
                .rename(columns={"subject_id": "participant_id"}))
    df = df.merge(
        feats[["participant_id", "SD_daily_mesor",
                "SD_daily_amplitude", "SD_daily_acrophase"]],
        on="participant_id", how="left",
    )
    return df


def run_primary_analysis(slug: str, outcome_label: str,
                          out_filename: str) -> dict:
    """Run the full primary-analysis stack for one outcome and persist results.

    Saves: TABLES_DIR / out_filename     (TSV with one row per model)
           OUTPUTS_DIR / out_filename.replace('.tsv', '.log')

    Returns the results dict (used by figures or downstream callers).
    """
    out_lines: list[str] = []
    def log(msg: str = ""):
        print(msg); out_lines.append(msg)

    df = load_analytic_frame(slug)
    n = len(df); n_case = int(df["onset"].sum())
    pct = 100 * n_case / n
    log("=" * 78)
    log(f"Primary analyses · {outcome_label}")
    log("=" * 78)
    log(f"  Analytic cohort: n = {n:,}, incident cases = {n_case} ({pct:.1f}%)")

    # ----- Between-person -----
    rows_btw: list[dict] = []
    log("\n--- Between-person analyses ---")
    log("  (no multiple-comparison correction across the three rhythm parameters)")
    for col, label in BETWEEN:
        r = fit_logistic_cluster(df, [col], return_predictor=col)
        rows_btw.append({"label": label, "predictor": col,
                          "n": r.n, "n_cases": r.n_cases,
                          "OR": r.OR, "OR_lo": r.OR_lo,
                          "OR_hi": r.OR_hi, "p": r.p})
        log(f"  {label:<24s}  {fmt_or(r)}")

    # ----- Within-person + joint -----
    rows_wpu: list[dict] = []
    rows_wpj: list[dict] = []
    rows_msame: list[dict] = []
    log("\n--- Within-person analyses ---")
    for col, label in WITHIN:
        sub = df.dropna(subset=[col]).copy()
        u = fit_logistic_cluster(sub, [col], return_predictor=col)
        joint = fit_logistic_cluster(sub, [col, "typical_day_mesor"])
        m_same = fit_logistic_cluster(sub, ["typical_day_mesor"],
                                       return_predictor="typical_day_mesor")
        if u is None:
            continue
        rows_wpu.append({"label": label, "predictor": col,
                          "n": u.n, "n_cases": u.n_cases,
                          "OR": u.OR, "OR_lo": u.OR_lo, "OR_hi": u.OR_hi,
                          "p": u.p})
        if joint is not None:
            jp = joint[col]; jm = joint["typical_day_mesor"]
            rows_wpj.append({"label": label, "predictor": col,
                              "n": jp.n, "n_cases": jp.n_cases,
                              "OR_feature": jp.OR,
                              "OR_feature_lo": jp.OR_lo,
                              "OR_feature_hi": jp.OR_hi,
                              "p_feature": jp.p,
                              "OR_mesor_in_joint": jm.OR,
                              "p_mesor_in_joint": jm.p})
        if m_same is not None:
            rows_msame.append({"label": label, "n": m_same.n,
                                "n_cases": m_same.n_cases,
                                "mesor_OR_sameN": m_same.OR,
                                "mesor_OR_sameN_lo": m_same.OR_lo,
                                "mesor_OR_sameN_hi": m_same.OR_hi,
                                "mesor_p_sameN": m_same.p})
        log(f"  {label:<24s}  n = {u.n}  cases = {u.n_cases}  "
            f"univariate {fmt_or(u)}")
        if joint is not None:
            jp = joint[col]; jm = joint["typical_day_mesor"]
            log(f"  {label:<24s}  joint with mesor: feature OR = {jp.OR:.2f} "
                f"[{jp.OR_lo:.2f}, {jp.OR_hi:.2f}], p = {jp.p:.3g}; "
                f"mesor OR = {jm.OR:.2f}, p = {jm.p:.3g}")

    # ----- Save -----
    btw_df   = pd.DataFrame(rows_btw);   btw_df["analysis"]   = "between"
    wpu_df   = pd.DataFrame(rows_wpu);   wpu_df["analysis"]   = "within_univariate"
    wpj_df   = pd.DataFrame(rows_wpj);   wpj_df["analysis"]   = "within_joint_with_mesor"
    msame_df = pd.DataFrame(rows_msame); msame_df["analysis"] = "mesor_same_N_re_run"
    out_path = TABLES_DIR / out_filename
    pd.concat([btw_df, wpu_df, wpj_df, msame_df], ignore_index=True).to_csv(
        out_path, sep="\t", index=False)
    log(f"\nWrote {out_path}")
    log_path = OUTPUTS_DIR / out_filename.replace(".tsv", ".log")
    log_path.write_text("\n".join(out_lines))
    print(f"Wrote {log_path}")

    return {
        "analytic_df": df,
        "between": rows_btw,
        "within_univariate": rows_wpu,
        "within_joint": rows_wpj,
        "mesor_same_N": rows_msame,
        "n": n, "n_cases": n_case, "n_controls": n - n_case,
    }
