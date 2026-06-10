#!/usr/bin/env python3
"""Parent KSADS diagnostic onset (Table S5): convergent check on the CBCL onset
result for the same three categories against the n=1,188 control pool. Categories
(ever met = present|past|partial_remission): Depression = dep; Anxiety = gad,
sepanx, socanx, panic, agor; Externalizing = adhd|odd|cond. Lenient vs
conservative eligibility; onset first met at Year 4/6 (correctness clause).
Model: onset ~ mesor + amplitude + acrophase + age + sex, family-clustered, per
1 SD. Control-invariance panel swaps in KSADS-all-clear controls.
"""
import os, sys, numpy as np, pandas as pd, statsmodels.api as sm

KS = "/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/ABCD/ABCD 7.0/KSADS"
RAW = f"{KS}/rawdata/phenotype"; OUT = f"{KS}/abcd_ksads/outputs"
SH = ("/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/ABCD/"
      "ABCD Actigraphy Resource Paper/fitbit_prediction_superhealthy")
REPO = ("/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/ABCD/"
        "ABCD Actigraphy Resource Paper/fitbit_prediction_github/code")
sys.path.insert(0, REPO)
from utils.outcomes import load_sex, load_family
from utils.paths import COSINOR_BLUP_W2
exec(open(f"{KS}/abcd_ksads/scripts/11_fitbit_ksads_models.py").read().split("# ---- build ANY")[0])

FULL = {"present", "past", "partial_remission"}
CATS = {"Depression": ["dep"],
        "Anxiety": ["gad", "sepanx", "socanx", "panic", "agor"],
        "Externalizing": ["adhd", "odd", "cond"]}
UNION = sorted({m for ms in CATS.values() for m in ms})

# ---- ultra-clean CBCL HC ids (n = 1,188) ----
hc_cbcl = set(pd.read_csv(f"{SH}/results/tables/ultraclean_hc_ids.csv")["participant_id"])

# ---- covariates ----
bl = pd.read_parquet(COSINOR_BLUP_W2).rename(columns={"subject_id": "participant_id"})
sex = load_sex()[["participant_id", "is_female"]]; fam = load_family()
age = pd.read_csv(f"{RAW}/mh_p_ksads__dep.tsv", sep="\t",
                  usecols=["participant_id", "session_id", "mh_p_ksads__dep_age"], dtype=str)
age = age[age.session_id == "ses-02A"]
age["age_yrs"] = pd.to_numeric(age["mh_p_ksads__dep_age"], errors="coerce")
COV = (bl[["participant_id", "mesor_blup", "amplitude_blup", "acrophase_blup"]]
       .merge(age[["participant_id", "age_yrs"]], on="participant_id", how="left")
       .merge(sex, on="participant_id", how="left")
       .merge(fam, on="participant_id", how="left"))
for c in ["mesor_blup", "amplitude_blup", "acrophase_blup"]:
    COV[c + "_z"] = (COV[c] - COV[c].mean()) / COV[c].std()
FZ = ["mesor_blup_z", "amplitude_blup_z", "acrophase_blup_z"]
BASE = ["age_yrs", "is_female"]

# ---- KSADS-all-clear controls (healthy on all 3 categories at every wave) ----
uf = wide_indicator(UNION, FULL)
for w in ("W1", "W2", "W3", "W4"):
    if w not in uf: uf[w] = np.nan
hc_ksads = set(uf.index[(uf["W1"] == 0) & (uf["W2"] == 0) & (uf["W3"] == 0) & (uf["W4"] == 0)])

# union eligibility (for conservative case ascertainment)
union_elig = (uf["W1"] == 0) & (uf["W2"] == 0)

log = []
def P(*a): s = " ".join(map(str, a)); print(s); log.append(s)

def run(cases, hc_ids, tag):
    ids = cases | hc_ids
    d = pd.DataFrame({"participant_id": list(ids)})
    d["y"] = d.participant_id.isin(cases).astype(float)
    d = d.merge(COV, on="participant_id", how="inner").dropna(subset=FZ + BASE + ["family_id"])
    d = d[~((d.y == 0) & d.participant_id.isin(cases))]
    f = sm.Logit(d["y"], sm.add_constant(d[FZ + BASE].astype(float))).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": d["family_id"]}, maxiter=300)
    ors = "  ".join(
        f"{t.split('_')[0]}={np.exp(f.params[t]):.2f}[{np.exp(f.conf_int().loc[t][0]):.2f},"
        f"{np.exp(f.conf_int().loc[t][1]):.2f}]{'*' if f.pvalues[t] < .05 else ''}(p={f.pvalues[t]:.3f})"
        for t in FZ)
    P(f"    {tag:34} cases={int(d['y'].sum()):>4}  HC={int((d['y']==0).sum()):>4}  {ors}")

for cat, mods in CATS.items():
    P(f"\n### {cat}  ({'+'.join(mods)})")
    kf = wide_indicator(mods, FULL)
    # lenient: eligibility from this category only
    s_len = incident_series(kf.copy(), kf.copy())
    cases_len = set(s_len.index[(s_len["elig"] == True) & (s_len["inc"] == 1)])
    # conservative: eligibility from the union of all three categories
    s_con = incident_series(kf.copy(), uf.copy())
    cases_con = set(s_con.index[(s_con["elig"] == True) & (s_con["inc"] == 1)])
    P("  Controls = CBCL ultra-clean HC (n=1,188):")
    run(cases_len, hc_cbcl, "lenient   (this-category eligible)")
    run(cases_con, hc_cbcl, "conservative (all-clear eligible)")
    P("  Control-invariance: KSADS-all-clear HC:")
    run(cases_len, hc_ksads, "lenient   (this-category eligible)")
    run(cases_con, hc_ksads, "conservative (all-clear eligible)")

open(f"{OUT}/19_ksads_threecat_hc.log", "w").write("\n".join(log))
print(f"\nSaved -> outputs/19_ksads_threecat_hc.log")
