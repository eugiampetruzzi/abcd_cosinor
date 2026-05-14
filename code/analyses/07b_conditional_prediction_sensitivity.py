"""07b · Conditional prediction — Design A (SENSITIVITY).

Stricter cohort restriction than 07a: anchor first-incident at Wave 1 only
(not Wave 1 OR Wave 2). All other definitions identical to 07a.

Run for sensitivity comparison: under stricter temporal ordering (rhythm
genuinely measured *after* anchor onset rather than potentially concurrent),
do the prediction effects hold?

At-risk cohort (per anchor × target):
    - anchor_first_wave == W1                  (anchor present at baseline)
    - target observed and == 0 at W2
    - target observed at least once at W3 or W4

Outputs:
    results/tables/conditional_prediction_sensitivity.tsv
    results/outputs/07b_conditional_prediction_sensitivity.log
"""
from __future__ import annotations
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cooccurrence import load_cooccurrence_frame  # noqa: E402
from utils.paths import TABLES_DIR, OUTPUTS_DIR, W1, W2, W3, W4  # noqa: E402
from utils.modeling import fit_logistic_cluster  # noqa: E402

TABLES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = TABLES_DIR / "conditional_prediction_sensitivity.tsv"
LOG_PATH = OUTPUTS_DIR / "07b_conditional_prediction_sensitivity.log"

ANCHORS = [
    ("dep",     "depression",   "dep_first_wave"),
    ("obesity", "obesity",      "obesity_first_wave"),
    ("htn",     "hypertension", "htn_first_wave"),
]
TARGETS = [
    ("dep",     "depression",   "dep_first_wave",     "dep_at_w2",     "dep_obs_w3w4"),
    ("obesity", "obesity",      "obesity_first_wave", "obesity_at_w2", "obesity_obs_w3w4"),
    ("htn",     "hypertension", "htn_first_wave",     "htn_at_w2",     "htn_obs_w3w4"),
]
PREDICTORS = [
    ("cosinor_mesor_w2",     "mesor"),
    ("cosinor_amplitude_w2", "amplitude"),
    ("cosinor_acrophase_w2", "acrophase"),
    ("sd_daily_mesor",       "SD daily mesor"),
    ("sd_daily_amplitude",   "SD daily amplitude"),
    ("sd_daily_acrophase",   "SD daily acrophase"),
]


def at_risk_design_A(df: pd.DataFrame, anchor_wave_col: str,
                      target_at_w2: str, target_first_wave: str,
                      target_obs_w3w4: str) -> pd.DataFrame:
    return df[(df[anchor_wave_col] == W1)
               & (df[target_at_w2] == 0)
               & (df[target_obs_w3w4])].copy()


def main() -> None:
    df = load_cooccurrence_frame().rename(columns={"age_w2": "age_yrs"})
    print(f"Cosinor-pass cohort: n = {len(df):,}")
    print("DESIGN A (SENSITIVITY): anchor positive at W1 ONLY, target negative at W2,")
    print("                          target observed at W3 or W4.")
    print()
    print("Cell sizes:")

    rows: list[dict] = []
    cells_for_log: list[tuple] = []

    for anchor_short, anchor_label, anchor_wave_col in ANCHORS:
        for tgt_short, tgt_label, tgt_first, tgt_w2, tgt_obs in TARGETS:
            if anchor_short == tgt_short:
                continue
            sub = at_risk_design_A(df, anchor_wave_col, tgt_w2, tgt_first, tgt_obs)
            sub["onset"] = sub[tgt_first].isin([W3, W4]).astype(int)
            n_at_risk = len(sub)
            n_events = int(sub["onset"].sum())
            flag = ""
            if n_at_risk == 0:
                flag = "  <-- empty cell"
            elif n_events < 20:
                flag = "  <-- underpowered (events < 20)"
            print(f"  Anchor={anchor_label:<13s} -> Target={tgt_label:<13s}: "
                  f"n_at_risk = {n_at_risk:>5d}, n_events = {n_events:>4d}{flag}")
            cells_for_log.append((anchor_label, tgt_label, n_at_risk, n_events))

            if n_at_risk == 0 or n_events == 0:
                for col, plabel in PREDICTORS:
                    rows.append({
                        "anchor_condition": anchor_label,
                        "target_condition": tgt_label,
                        "predictor": plabel,
                        "or_per_sd": float("nan"),
                        "ci_lo": float("nan"), "ci_hi": float("nan"),
                        "p": float("nan"),
                        "n_at_risk": n_at_risk, "n_events": n_events,
                        "note": "empty cell" if n_at_risk == 0 else "no events",
                    })
                continue

            for col, plabel in PREDICTORS:
                r = fit_logistic_cluster(sub, [col], return_predictor=col)
                if r is None:
                    rows.append({
                        "anchor_condition": anchor_label,
                        "target_condition": tgt_label,
                        "predictor": plabel,
                        "or_per_sd": float("nan"),
                        "ci_lo": float("nan"), "ci_hi": float("nan"),
                        "p": float("nan"),
                        "n_at_risk": n_at_risk, "n_events": n_events,
                        "note": "model failed",
                    })
                    continue
                rows.append({
                    "anchor_condition": anchor_label,
                    "target_condition": tgt_label,
                    "predictor": plabel,
                    "or_per_sd": r.OR,
                    "ci_lo": r.OR_lo, "ci_hi": r.OR_hi, "p": r.p,
                    "n_at_risk": r.n, "n_events": r.n_cases,
                    "note": "" if n_events >= 20 else "underpowered",
                })

    print()
    print("Per-cell results (all three cosinor predictors side-by-side):")
    out = pd.DataFrame(rows)
    for (a, t, _, _) in cells_for_log:
        cell_rows = out[(out["anchor_condition"] == a)
                          & (out["target_condition"] == t)]
        if cell_rows.empty:
            continue
        n_ar = int(cell_rows["n_at_risk"].iloc[0])
        n_ev = int(cell_rows["n_events"].iloc[0])
        print(f"\n  {a} → {t}  (n_at_risk = {n_ar:,}, n_events = {n_ev})")
        for _, r in cell_rows.iterrows():
            if pd.isna(r["or_per_sd"]):
                print(f"    {r['predictor']:<10s}  <{r['note']}>")
            else:
                sig = " *" if r["p"] < 0.05 else ""
                print(f"    {r['predictor']:<10s}  OR = {r['or_per_sd']:.2f} "
                      f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}], "
                      f"p = {r['p']:.3g}{sig}")

    out.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
