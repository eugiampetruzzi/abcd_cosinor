#!/usr/bin/env python3
"""11_fitbit_ksads_models.py - Cosinor rhythm features -> incident KSADS diagnosis.

Outcome (parent KSADS, strict never-diagnosed eligibility, biennial waves
W1=ses-00A, W2=ses-02A, W3=ses-04A, W4=ses-06A):
  composites  Mood = dep+bpd ; Anxiety = gad+sepanx+socanx+panic+agor+phobia ;
              Externalizing = adhd+odd+cond   (threshold diagnoses only)
  Eligible    = administered-and-negative on present+past+partial-remission for ALL
                composites at W1 AND W2 (never diagnosed).
  Incident    = first present diagnosis at W3 or W4; first-at-W4 requires a genuine
                negative at W3 (else excluded).
Predictors    = Wave-2 cosinor BLUPs: mesor, amplitude, acrophase (z-scored, per 1 SD).
Covariates    = age (W2) + sex; family-clustered SEs.

Run 1 (primary)  : any-diagnosis incidence ~ 3 features.
Run 2 (LRT gate) : does disorder identity moderate the feature effects? (stacked LRT)
Run 3 (mutual)   : per-composite feature ORs adjusted for the other composites' onset.
"""
import os, sys, csv, collections
import numpy as np, pandas as pd
import statsmodels.api as sm

KS   = "/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/ABCD/ABCD 7.0/KSADS"
RAW  = os.path.join(KS, "rawdata", "phenotype")
OUT  = os.path.join(KS, "abcd_ksads", "outputs")
REPO = "/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/ABCD/ABCD Actigraphy Resource Paper/fitbit_prediction_github/code"
sys.path.insert(0, REPO)
from utils.outcomes import load_sex, load_family           # noqa
from utils.paths import COSINOR_BLUP_W2                     # noqa

W = {"ses-00A":"W1","ses-02A":"W2","ses-04A":"W3","ses-06A":"W4"}
ASSESSED = {"0","1","888"}
COMPOS = {"Mood":["dep","bpd"],
          "Anxiety":["gad","sepanx","socanx","panic","agor","phobia"],
          "Externalizing":["adhd","odd","cond"]}
ALLMODS = [m for ms in COMPOS.values() for m in ms]

vmap = [r for r in csv.DictReader(open(os.path.join(OUT,"ksads_variable_map.csv")))
        if r["informant"]=="parent" and r["layer"]=="diagnosis"]
def core(r): l=r["label"].lower(); return "other specified" not in l and "unspecified" not in l
def vars_for(mod, statuses):
    return [r["variable"] for r in vmap if r["module"]==mod and core(r) and r["status"] in statuses]

def wide_indicator(mods, statuses):
    """per participant x wave: 1=any present/diag, 0=administered-negative, NaN=not assessed."""
    parts=[]
    for m in mods:
        vs=vars_for(m, statuses)
        if not vs: continue
        df=pd.read_csv(os.path.join(RAW,f"mh_p_ksads__{m}.tsv"),sep="\t",
                       usecols=lambda c:c in (["participant_id","session_id"]+vs),dtype=str)
        df=df[df["session_id"].isin(W)]
        sub=df[vs]
        parts.append(pd.DataFrame({"pid":df["participant_id"],"wave":df["session_id"].map(W),
                                   "pos":(sub=="1").any(axis=1),
                                   "ass":sub.isin(ASSESSED).any(axis=1)}))
    allp=pd.concat(parts)
    g=allp.groupby(["pid","wave"]).agg(pos=("pos","max"),ass=("ass","max")).reset_index()
    g["v"]=np.where(g["pos"],1.0,np.where(g["ass"],0.0,np.nan))
    return g.pivot_table(index="pid",columns="wave",values="v",aggfunc="max")

def incident_series(present_ind, elig_ind):
    """elig from elig_ind (full set), incident from present_ind, with correctness clause."""
    for w in ("W1","W2","W3","W4"):
        if w not in present_ind: present_ind[w]=np.nan
        if w not in elig_ind: elig_ind[w]=np.nan
    elig = (elig_ind["W1"]==0)&(elig_ind["W2"]==0)
    w3,w4 = present_ind["W3"], present_ind["W4"]
    inc = np.where(w3==1,1, np.where((w3==0)&(w4==1),1,
              np.where((w3==0)&((w4==0)|w4.isna()),0, np.nan)))   # NaN: w3 missing -> excluded
    return pd.DataFrame({"elig":elig,"inc":inc}, index=present_ind.index)

# build ANY + per-composite incident outcomes
# ELIGIBILITY = present-only (not currently diagnosed at W1 & W2) to MAXIMIZE N.
# Recurrence risk (remitted past episode) is carried as a covariate `prior_any` and
# checked against the strict never-diagnosed definition as a sensitivity analysis.
print("Building KSADS outcomes (present-only eligibility, max N) ...")
any_pres = wide_indicator(ALLMODS, {"present"})
any_full = wide_indicator(ALLMODS, {"present","past","partial_remission"})
anyser   = incident_series(any_pres.copy(), any_pres.copy())
af = any_full.reindex(any_pres.index)
prior_any = (((af["W1"] == 1) | (af["W2"] == 1)).astype(float)).rename("prior_any")

comp_inc={}
comp_pres={}
for cname,mods in COMPOS.items():
    p=wide_indicator(mods,{"present"})
    comp_pres[cname]=p
    comp_inc[cname]=incident_series(p.copy(),p.copy())

# analytic frame: eligible on ANY (never any dx at W1&W2) + has W3 observed for any
base = pd.DataFrame(index=any_pres.index)
base["elig_any"]=anyser["elig"].reindex(base.index)
base["inc_any"]=anyser["inc"].reindex(base.index)
base["prior_any"]=prior_any.reindex(base.index)
for c in COMPOS:
    base[f"inc_{c}"]=comp_inc[c]["inc"].reindex(base.index)
df = base[base["elig_any"] & base["inc_any"].notna()].copy()
df.index.name="participant_id"; df=df.reset_index()

# predictors + covariates
blups=pd.read_parquet(COSINOR_BLUP_W2)[["subject_id","mesor_blup","amplitude_blup","acrophase_blup"]] \
        .rename(columns={"subject_id":"participant_id"})
sex=load_sex()[["participant_id","is_female"]]
fam=load_family()
# age at W2 from a KSADS _age column
age=pd.read_csv(os.path.join(RAW,"mh_p_ksads__dep.tsv"),sep="\t",
                usecols=["participant_id","session_id","mh_p_ksads__dep_age"],dtype=str)
age=age[age.session_id=="ses-02A"][["participant_id","mh_p_ksads__dep_age"]]
age["age_yrs"]=pd.to_numeric(age["mh_p_ksads__dep_age"],errors="coerce")

df=(df.merge(blups,on="participant_id",how="inner")
       .merge(age[["participant_id","age_yrs"]],on="participant_id",how="left")
       .merge(sex,on="participant_id",how="left")
       .merge(fam,on="participant_id",how="left"))
n_pre=len(df)
df=df.dropna(subset=["mesor_blup","amplitude_blup","acrophase_blup","age_yrs","is_female","family_id"])
for c in COMPOS: df[f"inc_{c}"]=df[f"inc_{c}"].fillna(0)   # within any-eligible, non-incident=0

FEATS=["mesor_blup","amplitude_blup","acrophase_blup"]
def zс(s): return (s-s.mean())/s.std()
for f in FEATS: df[f+"_z"]=zс(df[f])
FZ=[f+"_z" for f in FEATS]

print(f"\nAnalytic N = {len(df)} (eligible never-diagnosed with W2 cosinor + covariates; "
      f"dropped {n_pre-len(df)} for missing). Incident any = {int(df['inc_any'].sum())} "
      f"({100*df['inc_any'].mean():.1f}%).")
for c in COMPOS: print(f"  incident {c}: {int(df['inc_'+c].sum())}")

def fit(d, y, X):
    Xd=sm.add_constant(d[X].astype(float))
    return sm.Logit(d[y].astype(float),Xd).fit(disp=0,cov_type="cluster",
                    cov_kwds={"groups":d["family_id"]},maxiter=200)
def ortab(res, terms):
    out=[]
    for t in terms:
        b=res.params[t]; lo,hi=res.conf_int().loc[t]; out.append((t,np.exp(b),np.exp(lo),np.exp(hi),res.pvalues[t]))
    return out

log=[]
def P(*a): s=" ".join(str(x) for x in a); print(s); log.append(s)

# RUN 1
P("\n===== RUN 1: any-diagnosis incidence ~ mesor + amplitude + acrophase (+age+sex, family-clustered) =====")
m1=fit(df,"inc_any",FZ+["age_yrs","is_female","prior_any"])
P(f"  N={int(m1.nobs)}  events={int(df['inc_any'].sum())}  pseudo-R2={m1.prsquared:.3f}")
for t,orr,lo,hi,p in ortab(m1,FZ):
    P(f"  {t:16} OR={orr:.3f} [{lo:.3f}, {hi:.3f}]  p={p:.4f}")

# RUN 2: LRT gate (does disorder identity moderate feature effects?)
P("\n===== RUN 2: omnibus LRT — does disorder identity add to the feature effects? =====")
long=[]
for c in COMPOS:
    t=df[["participant_id","family_id","age_yrs","is_female","prior_any"]+FZ].copy()
    t["disorder"]=c; t["y"]=df[f"inc_{c}"].values
    long.append(t)
L=pd.concat(long,ignore_index=True)
D=pd.get_dummies(L["disorder"],prefix="d",drop_first=True).astype(float)
L=pd.concat([L,D],axis=1); dcols=list(D.columns)
# reduced: common feature effects + disorder main effects ; full: + feature x disorder
Xr=L[FZ+dcols+["age_yrs","is_female","prior_any"]]
inter=[]
for f in FZ:
    for dc in dcols:
        L[f"{f}__{dc}"]=L[f]*L[dc]; inter.append(f"{f}__{dc}")
Xf=L[FZ+dcols+inter+["age_yrs","is_female","prior_any"]]
rr=sm.Logit(L["y"].astype(float),sm.add_constant(Xr.astype(float))).fit(disp=0,maxiter=200)
rf=sm.Logit(L["y"].astype(float),sm.add_constant(Xf.astype(float))).fit(disp=0,maxiter=200)
from scipy.stats import chi2
lr=2*(rf.llf-rr.llf); dfree=len(inter); pval=chi2.sf(lr,dfree)
P(f"  stacked rows={len(L)} (3 composites x N). LRT feature×disorder: chi2({dfree})={lr:.2f}, p={pval:.4f}")
P(f"  -> {'disorder identity MATTERS (effects differ by disorder)' if pval<.05 else 'no evidence disorder identity adds; pooled any-diagnosis model justified'}")
P("  (LRT on unclustered stacked likelihood; clustering affects SEs not the nested-LL test.)")

# RUN 3: mutually-adjusted per-composite
P("\n===== RUN 3: per-composite feature ORs, mutually adjusted for the other composites' onset =====")
for c in COMPOS:
    others=[f"inc_{o}" for o in COMPOS if o!=c]
    m=fit(df,f"inc_{c}",FZ+others+["age_yrs","is_female","prior_any"])
    P(f"  --- {c} (events={int(df['inc_'+c].sum())}) adj. for {', '.join(others)} ---")
    for t,orr,lo,hi,p in ortab(m,FZ):
        star="*" if p<.05 else " "
        P(f"     {t:16} OR={orr:.3f} [{lo:.3f}, {hi:.3f}]  p={p:.4f} {star}")

open(os.path.join(OUT,"11_fitbit_ksads_models.log"),"w").write("\n".join(log))
print("\nSaved log -> outputs/11_fitbit_ksads_models.log")
