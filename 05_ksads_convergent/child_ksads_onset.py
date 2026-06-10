#!/usr/bin/env python3
"""20_ksads_child_threecat_hc.py - CHILD (youth-report) KSADS onset, three categories.

Mirrors 19_ksads_threecat_hc.py but uses the youth self-report KSADS-COMP.
Youth KSADS assesses fewer modules than parent KSADS, so the categories are:
  Depression    = dep
  Anxiety       = gad, socanx, panic           (no separation/agoraphobia/phobia)
  Externalizing = cond                          (youth KSADS has no ADHD or ODD)

Controls (fixed) = CBCL ultra-clean HC (n = 1,188). Control-invariance panel uses
youth-KSADS-all-clear controls. Onset = first met criteria at Year 4 or 6,
clean at Baseline & Year 2; lenient vs conservative eligibility; per-1-SD ORs;
age + sex; family-clustered SEs.
"""
import os, sys, csv, collections, numpy as np, pandas as pd, statsmodels.api as sm

KS = "/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/ABCD/ABCD 7.0/KSADS"
RAW = f"{KS}/rawdata/phenotype"; OUT = f"{KS}/abcd_ksads/outputs"
SH = ("/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/ABCD/"
      "ABCD Actigraphy Resource Paper/fitbit_prediction_superhealthy")
REPO = ("/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/ABCD/"
        "ABCD Actigraphy Resource Paper/fitbit_prediction_github/code")
sys.path.insert(0, REPO)
from utils.outcomes import load_sex, load_family
from utils.paths import COSINOR_BLUP_W2

W = {"ses-00A": "W1", "ses-02A": "W2", "ses-04A": "W3", "ses-06A": "W4"}
ASSESSED = {"0", "1", "888"}
FULL = {"present", "past", "partial_remission"}
CATS = {"Depression": ["dep"],
        "Anxiety": ["gad", "socanx", "panic"],
        "Externalizing": ["cond"]}
UNION = sorted({m for ms in CATS.values() for m in ms})

# ---- youth diagnosis variable map ----
vmap = [r for r in csv.DictReader(open(f"{OUT}/ksads_variable_map.csv"))
        if r["informant"] == "youth" and r["layer"] == "diagnosis"]
def core(r): l = r["label"].lower(); return "other specified" not in l and "unspecified" not in l
def vars_for(mod, statuses):
    return [r["variable"] for r in vmap if r["module"] == mod and core(r) and r["status"] in statuses]

def wide_indicator(mods, statuses):
    parts = []
    for m in mods:
        vs = vars_for(m, statuses)
        if not vs: continue
        df = pd.read_csv(f"{RAW}/mh_y_ksads__{m}.tsv", sep="\t",
                         usecols=lambda c: c in (["participant_id", "session_id"] + vs), dtype=str)
        df = df[df["session_id"].isin(W)]
        sub = df[[c for c in vs if c in df.columns]]
        if sub.shape[1] == 0: continue
        parts.append(pd.DataFrame({"pid": df["participant_id"], "wave": df["session_id"].map(W),
                                   "pos": (sub == "1").any(axis=1),
                                   "ass": sub.isin(ASSESSED).any(axis=1)}))
    allp = pd.concat(parts)
    g = allp.groupby(["pid", "wave"]).agg(pos=("pos", "max"), ass=("ass", "max")).reset_index()
    g["v"] = np.where(g["pos"], 1.0, np.where(g["ass"], 0.0, np.nan))
    return g.pivot_table(index="pid", columns="wave", values="v", aggfunc="max")

def incident_series(present_ind, elig_ind):
    for w in ("W1", "W2", "W3", "W4"):
        if w not in present_ind: present_ind[w] = np.nan
        if w not in elig_ind: elig_ind[w] = np.nan
    elig = (elig_ind["W1"] == 0) & (elig_ind["W2"] == 0)
    w3, w4 = present_ind["W3"], present_ind["W4"]
    inc = np.where(w3 == 1, 1, np.where((w3 == 0) & (w4 == 1), 1,
                   np.where((w3 == 0) & ((w4 == 0) | w4.isna()), 0, np.nan)))
    return pd.DataFrame({"elig": elig, "inc": inc}, index=present_ind.index)

# ---- controls + covariates ----
hc_cbcl = set(pd.read_csv(f"{SH}/results/tables/ultraclean_hc_ids.csv")["participant_id"])
bl = pd.read_parquet(COSINOR_BLUP_W2).rename(columns={"subject_id": "participant_id"})
sex = load_sex()[["participant_id", "is_female"]]; fam = load_family()
age = pd.read_csv(f"{RAW}/mh_y_ksads__dep.tsv", sep="\t",
                  usecols=["participant_id", "session_id", "mh_y_ksads__dep_age"], dtype=str)
age = age[age.session_id == "ses-02A"]
age["age_yrs"] = pd.to_numeric(age["mh_y_ksads__dep_age"], errors="coerce")
COV = (bl[["participant_id", "mesor_blup", "amplitude_blup", "acrophase_blup"]]
       .merge(age[["participant_id", "age_yrs"]], on="participant_id", how="left")
       .merge(sex, on="participant_id", how="left").merge(fam, on="participant_id", how="left"))
for c in ["mesor_blup", "amplitude_blup", "acrophase_blup"]:
    COV[c + "_z"] = (COV[c] - COV[c].mean()) / COV[c].std()
FZ = ["mesor_blup_z", "amplitude_blup_z", "acrophase_blup_z"]; BASE = ["age_yrs", "is_female"]

uf = wide_indicator(UNION, FULL)
for w in ("W1", "W2", "W3", "W4"):
    if w not in uf: uf[w] = np.nan
hc_ksads = set(uf.index[(uf["W1"] == 0) & (uf["W2"] == 0) & (uf["W3"] == 0) & (uf["W4"] == 0)])

log = []
def P(*a): s = " ".join(map(str, a)); print(s); log.append(s)

def run(cases, hc_ids, tag):
    ids = cases | hc_ids
    d = pd.DataFrame({"participant_id": list(ids)})
    d["y"] = d.participant_id.isin(cases).astype(float)
    d = d.merge(COV, on="participant_id", how="inner").dropna(subset=FZ + BASE + ["family_id"])
    d = d[~((d.y == 0) & d.participant_id.isin(cases))]
    if int(d["y"].sum()) < 10:
        P(f"    {tag:34} cases={int(d['y'].sum()):>4}  (too few)"); return
    f = sm.Logit(d["y"], sm.add_constant(d[FZ + BASE].astype(float))).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": d["family_id"]}, maxiter=300)
    ors = "  ".join(
        f"{t.split('_')[0]}={np.exp(f.params[t]):.2f}[{np.exp(f.conf_int().loc[t][0]):.2f},"
        f"{np.exp(f.conf_int().loc[t][1]):.2f}]{'*' if f.pvalues[t] < .05 else ''}(p={f.pvalues[t]:.3f})"
        for t in FZ)
    P(f"    {tag:34} cases={int(d['y'].sum()):>4}  HC={int((d['y']==0).sum()):>4}  {ors}")

P("CHILD (youth-report) KSADS onset — three categories")
P("Depression=dep | Anxiety=gad,socanx,panic | Externalizing=cond (youth has no ADHD/ODD)")
for cat, mods in CATS.items():
    P(f"\n### {cat}  ({'+'.join(mods)})")
    kf = wide_indicator(mods, FULL)
    s_len = incident_series(kf.copy(), kf.copy())
    cases_len = set(s_len.index[(s_len["elig"] == True) & (s_len["inc"] == 1)])
    s_con = incident_series(kf.copy(), uf.copy())
    cases_con = set(s_con.index[(s_con["elig"] == True) & (s_con["inc"] == 1)])
    P("  Controls = CBCL ultra-clean HC (n=1,188):")
    run(cases_len, hc_cbcl, "lenient   (this-category eligible)")
    run(cases_con, hc_cbcl, "conservative (all-clear eligible)")
    P("  Control-invariance: youth-KSADS-all-clear HC:")
    run(cases_len, hc_ksads, "lenient   (this-category eligible)")
    run(cases_con, hc_ksads, "conservative (all-clear eligible)")

open(f"{OUT}/20_ksads_child_threecat_hc.log", "w").write("\n".join(log))
print(f"\nSaved -> outputs/20_ksads_child_threecat_hc.log")
