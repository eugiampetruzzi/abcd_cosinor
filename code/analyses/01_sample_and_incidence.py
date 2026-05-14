"""01 · Sample refinement and incident-diagnosis definition (common-HC framework).

In this build the healthy-control group is a single common pool used in every
primary analysis. The pool is restricted to participants who are clean on
all three primary conditions at every observed wave from baseline (Wave 1)
through follow-up (Wave 3 or Wave 4):

    Wave 1: clean on depression + obesity (BP not measured at baseline)
    Wave 2: clean on depression + obesity + hypertension (all three measured)
    Wave 3 or Wave 4: clean on all three at every observed follow-up wave
                       (if both observed, must be clean at both)

For each primary outcome, the analytic sample is the union of:
    (a) the common HC pool (onset = 0), and
    (b) participants in the pre-pool (clean on all three at W1 + W2) who
        develop the target outcome at Wave 3 or Wave 4 under the canonical
        first-onset rule.

The incident group does not need outcome data for all three conditions —
only the condition they developed.

Outputs:
    results/tables/analytic_depression.tsv
    results/tables/analytic_obesity.tsv
    results/tables/analytic_hypertension.tsv
    results/tables/sample_flow.tsv
    results/tables/incident_cases.tsv
    results/outputs/01_sample_and_incidence.log
"""
from __future__ import annotations
from pathlib import Path
import sys

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import (                          # noqa: E402
    TABLES_DIR, OUTPUTS_DIR, COSINOR_BLUP_W2,
    W1, W2, W3, W4,
)
from utils.outcomes import (                       # noqa: E402
    load_mental_health, load_physical_health, load_sex, load_family,
)
from utils.cohort import build_cohort              # noqa: E402

TABLES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def per_wave_status(mh: pd.DataFrame, phys: pd.DataFrame) -> pd.DataFrame:
    """Wide-format frame: one row per participant with dep/obesity/htn
    status at each wave (NaN if not observed)."""
    dep = (mh[["participant_id", "session_id", "dsm_dep_65"]]
              .dropna(subset=["dsm_dep_65"])
              .drop_duplicates(["participant_id", "session_id"])
              .pivot(index="participant_id", columns="session_id",
                      values="dsm_dep_65")
              .add_prefix("dep_"))
    obes = (phys[["participant_id", "session_id", "obese_85"]]
                .dropna(subset=["obese_85"])
                .drop_duplicates(["participant_id", "session_id"])
                .pivot(index="participant_id", columns="session_id",
                        values="obese_85")
                .add_prefix("obesity_"))
    htn = (phys[["participant_id", "session_id", "htn"]]
              .dropna(subset=["htn"])
              .drop_duplicates(["participant_id", "session_id"])
              .pivot(index="participant_id", columns="session_id",
                      values="htn")
              .add_prefix("htn_"))
    return dep.join(obes, how="outer").join(htn, how="outer").reset_index()


def _first_onset(s: dict, target_w3: str, target_w4: str) -> tuple[int, str | None]:
    """Canonical first-onset rule for a single outcome:
        W3 first onset:  W3 observed and == 1
        W4 first onset:  W4 observed and == 1 AND W3 observed and clean
        otherwise:       not an incident case
    """
    if target_w3 in s and s[target_w3] == 1:
        return 1, W3
    if target_w4 in s and s[target_w4] == 1:
        if target_w3 in s and s[target_w3] == 0:
            return 1, W4
    return 0, None


def build_common_hc_and_cases(status: pd.DataFrame) -> pd.DataFrame:
    """Apply the common-HC + per-outcome incident-case definitions.

    Returns a frame with columns:
        participant_id, in_prepool, is_common_hc,
        onset_dep, onset_obesity, onset_htn,
        first_onset_wave_dep, first_onset_wave_obesity, first_onset_wave_htn,
        any_incident
    """
    out = status.copy()

    # HCs (hybrid common pool): super-healthy at every observed wave +
    # per-outcome follow-up data availability.
    #   dep_ob_hc — super-healthy + has CBCL AND BMI at W3 or W4 (no BP req)
    #   htn_hc    — also has BP at W3 or W4  (⊂ dep_ob_hc)
    def _super_healthy_all_waves(row: pd.Series) -> int:
        for slug in ("dep", "obesity", "htn"):
            for w in (W1, W2, W3, W4):
                v = row.get(f"{slug}_{w}")
                if pd.notna(v) and v == 1:
                    return 0
        return 1

    def _has_followup(row: pd.Series, slug: str) -> bool:
        return any(pd.notna(row.get(f"{slug}_{w}")) for w in (W3, W4))

    def _dep_ob_hc(row: pd.Series) -> int:
        if not _super_healthy_all_waves(row):
            return 0
        return int(_has_followup(row, "dep") and _has_followup(row, "obesity"))

    def _htn_hc(row: pd.Series) -> int:
        if not _dep_ob_hc(row):
            return 0
        return int(_has_followup(row, "htn"))

    out["is_super_healthy"] = out.apply(_super_healthy_all_waves, axis=1)
    out["is_dep_ob_hc"]     = out.apply(_dep_ob_hc, axis=1)
    out["is_htn_hc"]        = out.apply(_htn_hc, axis=1)

    # Cases (canonical first-onset rule, unchanged from prior pipeline):
    # clean on TARGET outcome at its pre-waves + positive at W3 or W4.
    # Cross-condition status is NOT required for cases (only HCs use that).
    # Pre-waves: dep / obesity = (W1, W2); HTN = (W2,) since no W1 BP.
    pre_waves_per_outcome = {
        "dep":     (W1, W2),
        "obesity": (W1, W2),
        "htn":     (W2,),
    }

    def _outcome_dict(row: pd.Series, prefix: str) -> dict:
        d = {}
        for w in (W1, W2, W3, W4):
            v = row.get(f"{prefix}_{w}")
            if pd.notna(v):
                d[w] = int(v)
        return d

    def _case_for_outcome(s: dict, pre_waves: tuple) -> tuple[int, str | None]:
        """Canonical first-onset rule on TARGET outcome only."""
        # W2 must be observed and clean (predictor wave)
        if W2 not in s or s[W2] == 1:
            return 0, None
        # Other pre-waves clean if observed
        if any(w in s and s[w] == 1 for w in pre_waves if w != W2):
            return 0, None
        # Need ≥1 post-W2 follow-up of this outcome
        if W3 not in s and W4 not in s:
            return 0, None
        # W3 first onset
        if W3 in s and s[W3] == 1:
            return 1, W3
        # W4 first onset (requires W3 observed clean)
        if W4 in s and s[W4] == 1:
            if W3 in s and s[W3] == 0:
                return 1, W4
            return 0, None  # cannot confirm first onset without W3
        return 0, None

    onsets = {"dep": [], "obesity": [], "htn": []}
    waves  = {"dep": [], "obesity": [], "htn": []}
    for _, row in out.iterrows():
        for slug in ("dep", "obesity", "htn"):
            d = _outcome_dict(row, slug)
            on, w = _case_for_outcome(d, pre_waves_per_outcome[slug])
            onsets[slug].append(on); waves[slug].append(w)
    for slug in ("dep", "obesity", "htn"):
        out[f"onset_{slug}"] = onsets[slug]
        out[f"first_onset_wave_{slug}"] = waves[slug]
    out["any_incident"] = (
        (out["onset_dep"] == 1) | (out["onset_obesity"] == 1) | (out["onset_htn"] == 1)
    ).astype(int)
    return out


def build_analytic_frame(
    spec: pd.DataFrame,
    slug: str,
    hc_col: str,
    blups: pd.DataFrame,
    age: pd.DataFrame,
    sex_pd: pd.DataFrame,
    family: pd.DataFrame,
) -> pd.DataFrame:
    """For outcome `slug`, return analytic frame: HCs from the specified HC
    pool (onset=0) + incident cases for this outcome (onset=1)."""
    sub = spec[(spec[hc_col] == 1) | (spec[f"onset_{slug}"] == 1)].copy()
    sub["onset"] = (sub[f"onset_{slug}"] == 1).astype(int)
    sub["first_onset_wave"] = sub[f"first_onset_wave_{slug}"]
    sub["definition"] = f"common_hc_{hc_col}_strict_onset"
    sub = sub[["participant_id", "onset", "first_onset_wave", "definition"]]
    df = (sub.merge(blups, on="participant_id", how="inner")
              .merge(age,  on="participant_id", how="left")
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
    log("Step 1: Participant flow + per-wave outcome matrix")
    log("=" * 78)

    sex    = load_sex()
    family = load_family()
    mh     = load_mental_health()
    phys   = load_physical_health(sex=sex)
    cohort = build_cohort(mh, phys)

    flow_rows = [
        ("Wave-2 Novel Technologies sub-study enrolled", len(cohort["enrolled_w2"])),
        ("Met Fitbit data-quality criteria (4-of-4, ≥3 days)", len(cohort["wearable_pass"])),
        ("Has Wave-2 cosinor BLUP",                        len(cohort["wearable_blup"])),
        ("Has any outcome at W3 or W4 (CBCL or BMI or BP)", len(cohort["any_outcome_w3_or_w4"])),
        ("Analytic pool (wearable-pass ∩ has-outcome)",    len(cohort["analytic_pool"])),
    ]
    for lbl, n in flow_rows:
        log(f"  {lbl:<55s}  n = {n:>5,}")
    pd.DataFrame(flow_rows, columns=["step", "n"]).to_csv(
        TABLES_DIR / "sample_flow.tsv", sep="\t", index=False)

    # ----- BLUPs and age at W2 -----
    blups = (pl.read_parquet(COSINOR_BLUP_W2)
                .filter(pl.col("r_squared").is_not_null())
                .select(["subject_id", "mesor_blup",
                          "amplitude_blup", "acrophase_blup"])
                .rename({"subject_id": "participant_id"})
                .to_pandas())
    cosinor_ids = set(blups["participant_id"])

    # Unified W2 age: coalesce across available age sources (cbcl/anthr/bp)
    # so participants without one specific measurement at W2 still get an age.
    cbcl_w2  = (mh[mh["session_id"] == W2][["participant_id", "cbcl_age"]]
                  .drop_duplicates("participant_id"))
    anthr_w2 = (phys[phys["session_id"] == W2][["participant_id", "anthr_age"]]
                  .drop_duplicates("participant_id"))
    bp_w2    = (phys[phys["session_id"] == W2][["participant_id", "bp_age"]]
                  .drop_duplicates("participant_id"))
    age_w2 = (cbcl_w2.merge(anthr_w2, on="participant_id", how="outer")
                      .merge(bp_w2,    on="participant_id", how="outer"))
    age_w2["age_yrs"] = (age_w2[["cbcl_age", "anthr_age", "bp_age"]]
                            .bfill(axis=1).iloc[:, 0])
    age_w2 = age_w2[["participant_id", "age_yrs"]].dropna(subset=["age_yrs"])
    age_dep = age_obs = age_htn = age_w2

    # ----- Per-wave outcome matrix + common-HC / case definitions -----
    log("\n" + "=" * 78)
    log("Step 2: Build per-wave outcome matrix and apply common-HC framework")
    log("=" * 78)

    status = per_wave_status(mh, phys)
    # Restrict to cosinor-pass cohort
    status = status[status["participant_id"].isin(cosinor_ids)].reset_index(drop=True)
    log(f"  participants with cosinor BLUP: {len(status):,}")

    spec = build_common_hc_and_cases(status)
    n_superhc   = int(spec["is_super_healthy"].sum())
    n_dep_ob_hc = int(spec["is_dep_ob_hc"].sum())
    n_htn_hc    = int(spec["is_htn_hc"].sum())
    n_dep       = int(spec["onset_dep"].sum())
    n_obes      = int(spec["onset_obesity"].sum())
    n_htn       = int(spec["onset_htn"].sum())
    n_any_inc   = int(spec["any_incident"].sum())

    log(f"  super-healthy at every observed wave:           "
        f"n = {n_superhc:,}")
    log(f"  HC for depression/obesity (super-healthy + CBCL & BMI at W3/W4): "
        f"n = {n_dep_ob_hc:,}")
    log(f"  HC for hypertension (subset with BP at W3/W4):  "
        f"n = {n_htn_hc:,}")
    log(f"  incident cases (canonical first-onset, target-outcome cleanness only):")
    log(f"    dep = {n_dep},  obesity = {n_obes},  HTN = {n_htn}")
    log(f"  participants who developed ≥1 of the 3:  n = {n_any_inc:,}")
    log(f"  case overlap:")
    log(f"    dep ∩ obesity = "
        f"{int(((spec['onset_dep']==1)&(spec['onset_obesity']==1)).sum())}")
    log(f"    dep ∩ HTN     = "
        f"{int(((spec['onset_dep']==1)&(spec['onset_htn']==1)).sum())}")
    log(f"    obesity ∩ HTN = "
        f"{int(((spec['onset_obesity']==1)&(spec['onset_htn']==1)).sum())}")
    log(f"    all three     = "
        f"{int(((spec['onset_dep']==1)&(spec['onset_obesity']==1)&(spec['onset_htn']==1)).sum())}")

    # ----- Per-outcome analytic frames -----
    log("\n" + "=" * 78)
    log("Step 3: Build per-outcome analytic frames (common HC + cases)")
    log("=" * 78)

    # Single HC pool across all three outcomes: the HTN HC group (N = 2,004),
    # which is confirmed clean on all three conditions at every observed wave
    # including blood pressure at follow-up. Used uniformly for dep, obesity,
    # and hypertension analyses.
    rows: list[dict] = []
    for slug, label, age, hc_col in [
            ("dep",     "Depression",   age_dep, "is_htn_hc"),
            ("obesity", "Obesity",      age_obs, "is_htn_hc"),
            ("htn",     "Hypertension", age_htn, "is_htn_hc")]:
        analytic = build_analytic_frame(spec, slug, hc_col, blups, age, sex, family)
        n_a = len(analytic); n_c = int(analytic["onset"].sum())
        n_hc_in_frame = n_a - n_c
        n_w3 = int((analytic["first_onset_wave"] == W3).sum())
        n_w4 = int((analytic["first_onset_wave"] == W4).sum())
        log(f"\n--- {label} ---")
        log(f"  analytic n = {n_a:,}  cases = {n_c}  HCs = {n_hc_in_frame:,}  "
            f"(W3 first: {n_w3}, W4 first: {n_w4})")
        out_slug = {"dep": "depression",
                      "obesity": "obesity",
                      "htn": "hypertension"}[slug]
        out_path = TABLES_DIR / f"analytic_{out_slug}.tsv"
        analytic.to_csv(out_path, sep="\t", index=False)
        log(f"  wrote {out_path.name}")
        rows.append({
            "outcome": label,
            "analytic_n": n_a, "analytic_cases": n_c,
            "analytic_HCs": n_hc_in_frame,
            "analytic_W3_first": n_w3, "analytic_W4_first": n_w4,
        })

    pd.DataFrame(rows).to_csv(TABLES_DIR / "incident_cases.tsv",
                                sep="\t", index=False)
    log(f"\nWrote {TABLES_DIR / 'incident_cases.tsv'}")

    (OUTPUTS_DIR / "01_sample_and_incidence.log").write_text("\n".join(out_lines))
    log(f"Wrote {OUTPUTS_DIR / '01_sample_and_incidence.log'}")


if __name__ == "__main__":
    main()
