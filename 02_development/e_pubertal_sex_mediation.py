"""Paper 2 - PDS as mediator of the sex difference in rhythm trajectory.

Parallel framework to paper2_hormone_sex_mediation.py:
  Model A:  rhythm ~ cr(age_yrs, df=4) + is_female + (1|pid)        -> beta_c
  Model B:  rhythm ~ cr(age_yrs, df=4) + is_female + pds_z + (1|pid) -> beta_c', beta_b
  Path a:   pds_z ~ cr(age_yrs, df=4) + is_female + (1|pid)         -> beta_a
  Indirect = a * b; MC CI from 10,000 independent normal draws.

Mediator: youth-report PDS (`pds_youth`), z-scored within the analytic cohort.
Within-wave age-PDS correlation is ~0.35 (VIF ~1.14); pooled r=0.71 reflects
that PDS rises with age across waves. The "never both age and PDS" convention
is for trajectory-variance models; mediation requires both, so we include them
together and rely on the modest within-wave collinearity.

Parent-report PDS is run as sensitivity.

Outputs (derivatives/paper2_pds_sex_mediation/):
- attenuation.tsv
- mediation.tsv
- summary.md
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from patsy import dmatrix

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import DERIV  # noqa: E402

PARAMS = [("mesor_blup", "MESOR (bpm)"),
          ("amplitude_blup", "Amplitude (bpm)"),
          ("acrophase_blup", "Acrophase (h)")]
N_MC = 10000
RNG_SEED = 20260522

OUT = DERIV / "paper2_pds_sex_mediation"
OUT.mkdir(parents=True, exist_ok=True)


lf = pl.read_parquet(DERIV / "rhythm_trajectory" / "long_form.parquet").to_pandas()
pds = pl.read_parquet(DERIV / "paper2_diagnostics_pds" / "pds_continuous.parquet").to_pandas()
pds["session_id"] = pds["session_id"].astype(str)
lf = lf.merge(pds[["participant_id", "session_id", "pds_youth", "pds_parent"]],
              on=["participant_id", "session_id"], how="left")


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


def fit_mixed(y: np.ndarray, X: pd.DataFrame, groups: np.ndarray):
    md = sm.MixedLM(y, X, groups=groups)
    return md.fit(reml=True, method="lbfgs", maxiter=300)


def run_one(rhythm_col: str, rhythm_label: str,
             pds_col: str, pds_label: str, tier: str) -> dict:
    df = lf.dropna(subset=[rhythm_col, "age_yrs", "is_female",
                            "participant_id", pds_col]).copy()
    df["is_female"] = df["is_female"].astype(int)
    df["pds_z"] = zscore(df[pds_col])
    n_obs = len(df); n_pid = df["participant_id"].nunique()

    X_age = dmatrix("cr(age_yrs, df=4)", df, return_type="dataframe")
    X_A = X_age.copy()
    X_A["is_female"] = df["is_female"].to_numpy()

    X_B = X_A.copy()
    X_B["pds_z"] = df["pds_z"].to_numpy()

    y_rhythm = df[rhythm_col].to_numpy()
    y_pds = df["pds_z"].to_numpy()
    grp = df["participant_id"].to_numpy()

    res_A = fit_mixed(y_rhythm, X_A, grp)
    res_B = fit_mixed(y_rhythm, X_B, grp)
    res_a = fit_mixed(y_pds, X_A, grp)

    beta_c = float(res_A.fe_params["is_female"])
    se_c = float(res_A.bse["is_female"])
    p_c = float(res_A.pvalues["is_female"])

    beta_c_prime = float(res_B.fe_params["is_female"])
    se_c_prime = float(res_B.bse["is_female"])
    p_c_prime = float(res_B.pvalues["is_female"])

    beta_b = float(res_B.fe_params["pds_z"])
    se_b = float(res_B.bse["pds_z"])
    p_b = float(res_B.pvalues["pds_z"])

    beta_a = float(res_a.fe_params["is_female"])
    se_a = float(res_a.bse["is_female"])
    p_a = float(res_a.pvalues["is_female"])

    rng = np.random.default_rng(RNG_SEED)
    a_draws = rng.normal(beta_a, se_a, size=N_MC)
    b_draws = rng.normal(beta_b, se_b, size=N_MC)
    ab_draws = a_draws * b_draws
    indirect = float(np.median(ab_draws))
    indirect_lo = float(np.percentile(ab_draws, 2.5))
    indirect_hi = float(np.percentile(ab_draws, 97.5))
    indirect_p = float(2 * min((ab_draws > 0).mean(), (ab_draws < 0).mean()))

    prop_med = indirect / beta_c if beta_c != 0 else float("nan")
    prop_med_draws = ab_draws / beta_c
    prop_med_lo = float(np.percentile(prop_med_draws, 2.5))
    prop_med_hi = float(np.percentile(prop_med_draws, 97.5))

    attenuation_pct = (beta_c - beta_c_prime) / beta_c * 100 if beta_c != 0 else float("nan")

    return {
        "rhythm": rhythm_label, "mediator": pds_label, "tier": tier,
        "n_obs": n_obs, "n_pid": n_pid,
        "beta_a": beta_a, "se_a": se_a, "p_a": p_a,
        "beta_b": beta_b, "se_b": se_b, "p_b": p_b,
        "beta_c": beta_c, "se_c": se_c, "p_c": p_c,
        "beta_c_prime": beta_c_prime, "se_c_prime": se_c_prime, "p_c_prime": p_c_prime,
        "attenuation_pct": attenuation_pct,
        "indirect": indirect, "indirect_lo": indirect_lo,
        "indirect_hi": indirect_hi, "indirect_p_mc": indirect_p,
        "prop_mediated": prop_med, "prop_mediated_lo": prop_med_lo,
        "prop_mediated_hi": prop_med_hi,
    }


MEDIATORS = [("pds_youth", "PDS (youth)", "primary"),
             ("pds_parent", "PDS (parent)", "sensitivity")]
rows: list[dict] = []
for rhy_col, rhy_lab in PARAMS:
    for med_col, med_lab, tier in MEDIATORS:
        r = run_one(rhy_col, rhy_lab, med_col, med_lab, tier)
        rows.append(r)
        print(f"{rhy_lab:18s} / {med_lab:15s}  c={r['beta_c']:+.3f}  "
              f"c'={r['beta_c_prime']:+.3f}  a*b={r['indirect']:+.3f}  "
              f"prop={r['prop_mediated']*100:+.1f}%")

res_df = pd.DataFrame(rows)

atten_cols = ["rhythm", "mediator", "tier", "n_obs", "n_pid",
              "beta_c", "se_c", "p_c",
              "beta_c_prime", "se_c_prime", "p_c_prime",
              "attenuation_pct"]
res_df[atten_cols].to_csv(OUT / "attenuation.tsv", sep="\t", index=False)

med_cols = ["rhythm", "mediator", "tier", "n_obs", "n_pid",
            "beta_a", "se_a", "p_a",
            "beta_b", "se_b", "p_b",
            "beta_c", "beta_c_prime",
            "indirect", "indirect_lo", "indirect_hi", "indirect_p_mc",
            "prop_mediated", "prop_mediated_lo", "prop_mediated_hi"]
res_df[med_cols].to_csv(OUT / "mediation.tsv", sep="\t", index=False)


md = [
    "# Paper 2 - PDS as mediator of the sex difference in rhythm\n\n",
    "Mediator: pubertal stage (PDS), z-scored within analytic cohort. ",
    "Primary = youth-report; sensitivity = parent-report.\n\n",
    "Models (sex-pooled, age-smoothed; random intercept per participant):\n",
    "- Model A (total):   rhythm ~ cr(age, df=4) + is_female\n",
    "- Model B (direct):  rhythm ~ cr(age, df=4) + is_female + pds_z\n",
    "- Path a:            pds_z ~ cr(age, df=4) + is_female\n",
    "- Indirect = a * b; MC CI from 10,000 independent normal draws.\n",
    "- Proportion mediated = indirect / beta_c.\n\n",
    "**Convention note:** the standing 'never both age + PDS' rule applies to ",
    "trajectory-variance attribution. For mediation, PDS is the explicit mediator and must enter the model with age; within-wave age-PDS r is ~0.35 (VIF ~1.14).\n\n",
    "## Attenuation of sex coefficient (Model A -> Model B)\n\n",
    "| Rhythm | Mediator | Tier | N | beta_c (total) | beta_c' (direct) | atten % |\n",
    "|---|---|---|---|---|---|---|\n",
]
for r in rows:
    md.append(
        f"| {r['rhythm']} | {r['mediator']} | {r['tier']} | "
        f"{r['n_obs']:,} obs / {r['n_pid']:,} pid | "
        f"{r['beta_c']:+.3f} (p={r['p_c']:.2g}) | "
        f"{r['beta_c_prime']:+.3f} (p={r['p_c_prime']:.2g}) | "
        f"{r['attenuation_pct']:+.1f}% |\n"
    )
md += [
    "\n## Mediation (sex -> PDS -> rhythm)\n\n",
    "| Rhythm | Mediator | Tier | a (sex->PDS, z) | b (PDS->rhythm) | a*b indirect | 95% MC CI | prop mediated (95% CI) |\n",
    "|---|---|---|---|---|---|---|---|\n",
]
for r in rows:
    md.append(
        f"| {r['rhythm']} | {r['mediator']} | {r['tier']} | "
        f"{r['beta_a']:+.3f} (p={r['p_a']:.2g}) | "
        f"{r['beta_b']:+.3f} (p={r['p_b']:.2g}) | "
        f"{r['indirect']:+.3f} | "
        f"[{r['indirect_lo']:+.3f}, {r['indirect_hi']:+.3f}] | "
        f"{r['prop_mediated']*100:+.1f}% "
        f"[{r['prop_mediated_lo']*100:+.1f}%, {r['prop_mediated_hi']*100:+.1f}%] |\n"
    )
md.append("\n*Sign convention*: is_female = 1 (female) - 0 (male).\n")
(OUT / "summary.md").write_text("".join(md))
print(f"\nWrote {OUT}")
