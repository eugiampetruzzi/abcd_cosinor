"""Build the comprehensive Supplementary Materials Word document (APA tables).

Addresses every "see Supplement" pointer in the manuscript:
  S1  Sample comparison (prediction vs. non-prediction remainder)
  S2  Co-development: bivariate parallel-process LGC (7 outcomes)
  S3  Incremental prediction of onset, hierarchical M1->M4 (5 outcomes)
  S4  Transdiagnostic between- and within-person onset, FDR across 30 tests
  S5  KSADS-COMP convergent check (case definitions + control invariance)

APA formatting: bold "Table SN", italic title, sentence-case centered headers,
Note block, Arial 12.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats as st

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

CODE = Path("/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/"
            "1 - Data/ABCD/ABCD Actigraphy Resource Paper/fitbit_prediction_superhealthy/code")
sys.path.insert(0, str(CODE))
from utils.paths import (ONEDRIVE, TABLES_DIR, RESULTS_DIR, MH_OUTCOMES,
                         PHYS_OUTCOMES, COSINOR_BLUP_W2, W1, W2, W3, W4)
from utils.outcomes import load_sex, load_family, load_physical_health

TRAJ = RESULTS_DIR / "trajectory"
OUT = Path("/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/"
           "1 - Data/ABCD/ABCD Actigraphy Resource Paper/Longitudinal Paper/combined paper")
DEMO_TSV = ONEDRIVE / "Release 6.1" / "Actigraphy_Eu_Outputs" / "subject_demographics.tsv"
KSLOG = ("/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/ABCD/"
         "ABCD 7.0/KSADS/abcd_ksads/outputs/19_ksads_threecat_hc.log")

FONT = "Arial"; SIZE = 12


# docx helpers
def _set_font(run, *, italic=False, bold=False, size=SIZE):
    run.font.name = FONT; run.font.size = Pt(size)
    run.font.italic = italic; run.font.bold = bold
    r = run._element.rPr.rFonts if run._element.rPr is not None else None
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts"); rpr.append(rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs"):
        rfonts.set(qn(a), FONT)


def _p(doc, runs, *, align=None, space_after=6, space_before=0):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    for text, kw in runs:
        _set_font(p.add_run(text), **kw)
    return p


def _cell_text(cell, text, *, italic=False, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=10):
    cell.text = ""
    para = cell.paragraphs[0]; para.alignment = align
    para.paragraph_format.space_after = Pt(2); para.paragraph_format.space_before = Pt(2)
    _set_font(para.add_run(text), italic=italic, bold=bold, size=size)


def _row_border(cell, side, sz=6):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders"); tcPr.append(borders)
    e = borders.find(qn(f"w:{side}"))
    if e is not None:
        borders.remove(e)
    e = OxmlElement(f"w:{side}")
    e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz))
    e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
    borders.append(e)


def add_table(doc, label, title, headers, rows, note, *,
              first_col_align=WD_ALIGN_PARAGRAPH.LEFT, tsize=10, group_rows=None):
    _p(doc, [(label, dict(bold=True))], space_after=0, space_before=10)
    _p(doc, [(title, dict(italic=True))], space_after=4)
    ncol = len(headers)
    tbl = doc.add_table(rows=1, cols=ncol)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = True
    hdr = tbl.rows[0].cells
    for j, h in enumerate(headers):
        _cell_text(hdr, h if False else hdr and h, size=tsize) if False else None
        _cell_text(hdr[j], h, bold=False, size=tsize,
                   align=WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER)
        _row_border(hdr[j], "top"); _row_border(hdr[j], "bottom")
    group_rows = group_rows or {}
    for i, row in enumerate(rows):
        cells = tbl.add_row().cells
        if i in group_rows:  # subheading spanning row (italic, left)
            pass
        for j, val in enumerate(row):
            al = first_col_align if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            ital = (j == 0 and i in group_rows)
            _cell_text(cells[j], val, italic=ital, size=tsize, align=al)
    for c in tbl.rows[-1].cells:
        _row_border(c, "bottom")
    # note
    np_ = doc.add_paragraph(); np_.paragraph_format.space_before = Pt(3)
    _set_font(np_.add_run("Note. "), italic=True, size=10)
    _set_font(np_.add_run(note), size=10)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def orci(o, lo, hi, p):
    return f"{o:.2f} [{lo:.2f}, {hi:.2f}]{stars(p)}"


# S1 data
def build_s1():
    mh = pd.read_parquet(MH_OUTCOMES)
    phys = pd.read_parquet(PHYS_OUTCOMES)
    demo = pd.read_csv(DEMO_TSV, sep="\t")
    blups = (pl.read_parquet(COSINOR_BLUP_W2).filter(pl.col("r_squared").is_not_null())
             .select(["subject_id", "mesor_blup", "amplitude_blup", "acrophase_blup"])
             .rename({"subject_id": "participant_id"}).to_pandas())
    cosinor = set(blups.participant_id)
    full = cosinor  # Wave-2 cosinor cohort

    # prediction sample = union of 5 onset frames
    sex = load_sex(); fam = load_family(); phh = load_physical_health(sex=sex)
    base = pd.read_csv(TABLES_DIR / "analytic_depression.tsv", sep="\t")
    base_hc = set(base[base.onset == 0].participant_id)
    dsm = ["cbcl_dsm_dep_tscore", "cbcl_dsm_anx_tscore", "cbcl_dsm_adhd_tscore",
           "cbcl_dsm_somat_tscore", "cbcl_dsm_cond_tscore", "cbcl_dsm_opp_tscore"]
    ev = set()
    for c in dsm:
        s = mh[["participant_id", c]].dropna(); ev |= set(s[s[c] >= 65].participant_id)
    ultra = base_hc - ev

    def fo(s):
        if W2 not in s or s[W2] == 1: return None
        if W1 in s and s[W1] == 1: return None
        if W3 not in s and W4 not in s: return None
        if W3 in s and s[W3] == 1: return 1
        if W4 in s and s[W4] == 1: return 1 if (W3 in s and s[W3] == 0) else None
        return None

    def cases(col):
        sub = mh[["participant_id", "session_id", col]].dropna(subset=[col]).copy()
        sub["f"] = (sub[col] >= 65).astype(int); sub = sub[sub.participant_id.isin(cosinor)]
        return {p for p, g in sub.groupby("participant_id") if fo(dict(zip(g.session_id, g.f))) == 1}

    # Prediction sample = union of the eight outcome-specific frames (matches Table 1)
    pred = set()
    for c in dsm:  # six CBCL DSM scales examined separately
        fu = set(mh[mh.session_id.isin([W3, W4])].dropna(subset=[c]).participant_id)
        pred |= (ultra & fu) | cases(c)
    for slug in ["obesity", "hypertension"]:
        a = pd.read_csv(TABLES_DIR / f"analytic_{slug}.tsv", sep="\t")
        pred |= set(a[a.onset == 1].participant_id) | (ultra & set(a[a.onset == 0].participant_id))
    pred &= cosinor
    rest = full - pred

    mh_w2 = mh[mh.session_id == W2].drop_duplicates("participant_id")
    ph_w2 = phys[phys.session_id == W2].drop_duplicates("participant_id")
    D = (demo.merge(blups, on="participant_id", how="inner")
         .merge(mh_w2[["participant_id", "cbcl_dsm_dep_tscore", "cbcl_dsm_anx_tscore",
                       "cbcl_dsm_adhd_tscore", "cbcl_dsm_cond_tscore", "cbcl_dsm_opp_tscore"]],
                on="participant_id", how="left")
         .merge(ph_w2[["participant_id", "bmi", "bp_sys_mean", "bp_dia_mean"]],
                on="participant_id", how="left"))
    D["grp"] = np.where(D.participant_id.isin(pred), "pred",
                        np.where(D.participant_id.isin(rest), "rest", "out"))
    D = D[D.grp.isin(["pred", "rest"])].copy()

    def cont(col):
        a = D[D.grp == "pred"][col].dropna(); b = D[D.grp == "rest"][col].dropna()
        t, p = st.ttest_ind(a, b, equal_var=False)
        return (f"{a.mean():.2f} ({a.std():.2f})", f"{b.mean():.2f} ({b.std():.2f})",
                f"{p:.3f}" if p >= .001 else "<.001")

    def cat(col, levels):
        out = []
        pred_c = D[D.grp == "pred"][col].value_counts()
        rest_c = D[D.grp == "rest"][col].value_counts()
        tab = np.array([[pred_c.get(l, 0) for l in levels], [rest_c.get(l, 0) for l in levels]])
        _, p, _, _ = st.chi2_contingency(tab)
        np_, nr = pred_c.sum(), rest_c.sum()
        for l in levels:
            out.append((l, f"{pred_c.get(l,0):,} ({100*pred_c.get(l,0)/np_:.1f})",
                        f"{rest_c.get(l,0):,} ({100*rest_c.get(l,0)/nr:.1f})", ""))
        out[0] = (out[0][0], out[0][1], out[0][2], f"{p:.3f}" if p >= .001 else "<.001")
        return out

    n_pred = (D.grp == "pred").sum(); n_rest = (D.grp == "rest").sum()
    rows = []
    rows.append(("Age, Wave 2 (years)", *cont("age_02A")))
    fem = cat("sex_label", ["Female", "Male"])
    # female only row
    pf = D[D.grp == "pred"]["sex_label"].eq("Female"); rf = D[D.grp == "rest"]["sex_label"].eq("Female")
    _, pfp, _, _ = st.chi2_contingency(np.array([[pf.sum(), (~pf).sum()], [rf.sum(), (~rf).sum()]]))
    rows.append(("Female", f"{pf.sum():,} ({100*pf.mean():.1f})", f"{rf.sum():,} ({100*rf.mean():.1f})",
                 f"{pfp:.3f}" if pfp >= .001 else "<.001"))
    rows.append(("Race / ethnicity", "", "", ""))
    for r in cat("ethnrace_label", ["White", "Hispanic", "Black", "Asian", "Other"]):
        rows.append(("  " + r[0], r[1], r[2], r[3]))
    rows.append(("Household income", "", "", ""))
    for r in cat("income_3lvl_label", ["<50k", "50–100k", ">100k"]):
        lab = {"<50k": "  < $50,000", "50–100k": "  $50,000–$100,000",
               ">100k": "  > $100,000"}[r[0]]
        rows.append((lab, r[1], r[2], r[3]))
    rows.append(("Parental education", "", "", ""))
    edu_levels = ["<HS", "HS/GED", "Some college", "Bachelor's", "Grad/professional"]
    edu_disp = {"<HS": "  Less than HS", "HS/GED": "  HS / GED", "Some college": "  Some college",
                "Bachelor's": "  Bachelor's degree", "Grad/professional": "  Graduate / professional"}
    for r in cat("edu_cgs_label", edu_levels):
        rows.append((edu_disp[r[0]], r[1], r[2], r[3]))
    rows.append(("Wave 2 mesor (bpm)", *cont("mesor_blup")))
    rows.append(("Wave 2 amplitude (bpm)", *cont("amplitude_blup")))
    rows.append(("Wave 2 acrophase (clock hour)", *cont("acrophase_blup")))
    rows.append(("CBCL Depression T", *cont("cbcl_dsm_dep_tscore")))
    rows.append(("CBCL Anxiety T", *cont("cbcl_dsm_anx_tscore")))
    rows.append(("CBCL ADHD T", *cont("cbcl_dsm_adhd_tscore")))
    rows.append(("BMI (kg/m²)", *cont("bmi")))
    rows.append(("Systolic BP (mmHg)", *cont("bp_sys_mean")))
    rows.append(("Diastolic BP (mmHg)", *cont("bp_dia_mean")))
    return rows, n_pred, n_rest


# build doc
doc = Document()
style = doc.styles["Normal"]; style.font.name = FONT; style.font.size = Pt(SIZE)

_p(doc, [("Supplementary Materials", dict(bold=True, size=14))],
   align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
_p(doc, [("Cardiac rhythm development: A wearable device index of risk for physical "
          "and mental illness in adolescence", dict(italic=True))],
   align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)

# ---- S1 ----
s1_rows, n_pred, n_rest = build_s1()
add_table(
    doc, "Table S1",
    "Comparison of the Prediction Sample and the Remainder of the Full Cosinor Cohort",
    ["Characteristic", f"Prediction sample (n = {n_pred:,})",
     f"Remainder (n = {n_rest:,})", "p"],
    s1_rows,
    "Continuous variables are M (SD); categorical variables are n (%). The prediction "
    "sample is the union of the five outcome-specific onset frames (Aim 2); the remainder "
    "comprises Wave-2 cosinor-cohort members not entering any onset analysis. p values are "
    "from Welch t tests (continuous) and chi-square tests (categorical; test reported on the "
    "first level of each block). CBCL = Child Behavior Checklist; ADHD = Attention-Deficit/"
    "Hyperactivity Disorder; BMI = body mass index; BP = blood pressure.",
    tsize=9.5)

# ---- S2 co-development ----
cod = [
    ("Systolic blood pressure", .893, .0005, .072, .021, -.105, .124),
    ("Body mass index",         .364, 3.0e-14, .209, 1e-16, .129, 6.6e-7),
    ("Depression",              .223, .0218, .134, 2.6e-14, .002, .938),
    ("ADHD",                    .197, .0355, .152, 1e-16, -.001, .966),
    ("Anxiety",                 .071, .299, .074, 8.5e-5, .033, .155),
    ("ODD",                     .051, .548, .102, 3.6e-10, -.020, .511),
    ("Conduct",                -.003, .982, .131, 2.7e-9, -.013, .732),
]
s2_rows = []
for lab, ssr, ssp, iir, iip, cb, cp in cod:
    s2_rows.append((lab, f"{ssr:.2f}{stars(ssp)}", f"{iir:.2f}{stars(iip)}",
                    f"{cb:+.2f}{stars(cp)}"))
add_table(
    doc, "Table S2",
    "Co-Development of Mesor and Health-Outcome Trajectories: Bivariate Parallel-Process Latent Growth Models",
    ["Outcome", "Slope-slope r", "Intercept-intercept r", "Cross-domain β"],
    s2_rows,
    "Bivariate parallel-process latent growth models estimating mesor and outcome trajectories "
    "jointly across Waves 2, 4, and 6 (MLR estimator, full-information maximum likelihood, "
    "site-clustered SEs), with sex predicting all growth factors. Slope-slope r = standardized "
    "covariance between mesor slope and outcome slope (positive = slower mesor decline tracks "
    "greater outcome increase); intercept-intercept r = concurrent level association; "
    "cross-domain β = Wave-2 mesor level predicting outcome slope. Outcomes are ordered by "
    "slope-slope effect size. ADHD = Attention-Deficit/Hyperactivity Disorder; ODD = "
    "Oppositional Defiant Disorder. * p < .05. ** p < .01. *** p < .001.")

# ---- S3 incremental ----
fit = pd.read_csv(TRAJ / "threecat_incremental_fit.csv")
lrt = pd.read_csv(TRAJ / "threecat_incremental_lrt.csv")
coef = pd.read_csv(TRAJ / "threecat_m4_coefficients.csv")
auc = fit.pivot(index="outcome", columns="model", values="auc")
ren = {"typical_day_mesor": "mesor", "typical_day_amplitude": "amplitude",
       "typical_day_acrophase": "acrophase"}
order = ["Depression", "Anxiety", "Externalizing", "Obesity", "Hypertension"]
s3_rows = []
for o in order:
    l34 = lrt[(lrt.outcome == o) & (lrt.full == "M4_+rhythm")].iloc[0]
    c = coef[coef.outcome == o].set_index("predictor")
    nev = int(c.iloc[0]["n_events"]); n = int(c.iloc[0]["n"])
    m = c.loc["typical_day_mesor"]; a = c.loc["typical_day_amplitude"]; ac = c.loc["typical_day_acrophase"]
    s3_rows.append((
        o, f"{nev}/{n}",
        f"{auc.loc[o,'M3_+behavior']:.3f}", f"{auc.loc[o,'M4_+rhythm']:.3f}",
        f"+{l34['delta_auc']:.3f}", f"{l34['p']:.3g}{stars(l34['p'])}",
        orci(m['or'], m['or_lo'], m['or_hi'], m['p']),
        orci(a['or'], a['or_lo'], a['or_hi'], a['p']),
        orci(ac['or'], ac['or_lo'], ac['or_hi'], ac['p'])))
add_table(
    doc, "Table S3",
    "Incremental Prediction of Clinical Onset by Wave-2 Cardiac Rhythm (Hierarchical M1–M4)",
    ["Outcome", "Cases/N", "AUC M3", "AUC M4", "ΔAUC", "LRT p",
     "Mesor", "Amplitude", "Acrophase"],
    s3_rows,
    "Rhythm cells are odds ratios [95% CI] per 1-SD from the fully adjusted model (M4). "
    "Hierarchical logistic models versus the healthy-control pool (n = 1,188), family-clustered "
    "SEs. Blocks: M1 demographics; M2 + baseline symptoms (Wave-0 T-score, or Wave-0 BMI / Wave-2 "
    "systolic BP for cardiometabolic outcomes); M3 + behavior (Wave-2 sleep, MVPA); M4 + rhythm. "
    "ΔAUC and the likelihood-ratio test (LRT) compare M4 with M3. Externalizing = onset on any of "
    "ADHD, ODD, or Conduct. AUC = area under the receiver operating characteristic curve. "
    "* p < .05. ** p < .01. *** p < .001.", tsize=9)

# ---- S4 between/within FDR-30 ----
bw = pd.read_csv(TRAJ / "threecat_between_within.csv")
def bw_block(analysis):
    rows = []
    for o in order:
        sub = bw[(bw.analysis == analysis) & (bw.outcome == o)].set_index("predictor")
        cells = [o]
        keys = (["Mesor", "Amplitude", "Acrophase"] if analysis == "between"
                else ["SD Mesor", "SD Amplitude", "SD Acrophase"])
        for k in keys:
            r = sub.loc[k]
            cells.append(f"{r['OR']:.2f} [{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]{stars(r['p_fdr'])}")
        rows.append(tuple(cells))
    return rows
s4_between = bw_block("between"); s4_within = bw_block("within")
# combine into one table with a spanning subheading
s4_rows = [("Between-person", "", "", "")]
s4_rows += s4_between
s4_rows += [("Within-person", "", "", "")]
s4_rows += s4_within
grp = {0, 1 + len(order)}
add_table(
    doc, "Table S4",
    "Transdiagnostic Specificity: Between- and Within-Person Rhythm Features Predicting Incident Onset",
    ["Outcome", "Mesor", "Amplitude", "Acrophase"],
    s4_rows,
    "Cells are odds ratios [95% CI] per 1-SD. Logistic models versus the healthy-control pool "
    "(n = 1,188), adjusting for age and sex with family-clustered SEs. Between-person predictors "
    "are the typical-day cosinor parameters; within-person predictors are the standard deviations "
    "of daily mesor, amplitude, and acrophase among participants with 7+ valid daily fits. "
    "Significance reflects Benjamini-Hochberg FDR correction across all 30 tests (5 outcomes × "
    "6 predictors). Estimates are those summarized in Figure 3. * p < .05. ** p < .01. "
    "*** p < .001.", tsize=9.5, group_rows=grp)

# ---- S5 KSADS ----
ks = {  # (cases, mesor, amp, acro) -> from 19_ksads_threecat_hc.log
    ("CBCL HC", "Depression", "Lenient"): (561, "1.50 [1.34, 1.69]***", "0.89 [0.79, 1.00]", "1.18 [1.04, 1.34]**"),
    ("CBCL HC", "Depression", "Conservative"): (216, "1.34 [1.14, 1.57]***", "0.87 [0.74, 1.03]", "1.17 [0.99, 1.38]"),
    ("CBCL HC", "Anxiety", "Lenient"): (414, "1.43 [1.25, 1.63]***", "0.85 [0.75, 0.97]*", "1.00 [0.88, 1.15]"),
    ("CBCL HC", "Anxiety", "Conservative"): (259, "1.34 [1.15, 1.56]***", "0.81 [0.70, 0.95]**", "1.08 [0.92, 1.25]"),
    ("CBCL HC", "Externalizing", "Lenient"): (210, "1.41 [1.19, 1.68]***", "1.06 [0.90, 1.24]", "1.05 [0.87, 1.26]"),
    ("CBCL HC", "Externalizing", "Conservative"): (157, "1.37 [1.13, 1.66]**", "1.07 [0.89, 1.29]", "1.01 [0.81, 1.24]"),
    ("KSADS HC", "Depression", "Lenient"): (561, "1.24 [1.12, 1.37]***", "1.04 [0.94, 1.15]", "1.02 [0.93, 1.13]"),
    ("KSADS HC", "Depression", "Conservative"): (216, "1.09 [0.94, 1.25]", "1.00 [0.86, 1.17]", "1.02 [0.89, 1.16]"),
    ("KSADS HC", "Anxiety", "Lenient"): (414, "1.15 [1.03, 1.28]*", "0.99 [0.88, 1.11]", "0.91 [0.82, 1.01]"),
    ("KSADS HC", "Anxiety", "Conservative"): (259, "1.08 [0.95, 1.23]", "0.93 [0.80, 1.07]", "0.96 [0.85, 1.08]"),
    ("KSADS HC", "Externalizing", "Lenient"): (210, "1.18 [1.02, 1.37]*", "1.22 [1.05, 1.42]*", "0.94 [0.82, 1.08]"),
    ("KSADS HC", "Externalizing", "Conservative"): (157, "1.15 [0.97, 1.36]", "1.23 [1.03, 1.47]*", "0.92 [0.78, 1.08]"),
}
s5_rows = []
s5_rows.append(("CBCL healthy controls (n = 1,188)", "", "", "", ""))
gr = {0}
for cat_ in ["Depression", "Anxiety", "Externalizing"]:
    c, m, a, ac = ks[("CBCL HC", cat_, "Lenient")]
    s5_rows.append((cat_, str(c), m, a, ac))
s5_rows.append(("KSADS healthy controls (n = 2,587)", "", "", "", ""))
gr.add(len(s5_rows) - 1)
for cat_ in ["Depression", "Anxiety", "Externalizing"]:
    c, m, a, ac = ks[("KSADS HC", cat_, "Lenient")]
    s5_rows.append((cat_, str(c), m, a, ac))
add_table(
    doc, "Table S5",
    "KSADS-COMP Diagnostic Onset Predicted by Wave-2 Cardiac Rhythm (Exploratory Convergent Check)",
    ["Category", "Cases", "Mesor", "Amplitude", "Acrophase"],
    s5_rows,
    "Cells are odds ratios [95% CI] per 1-SD from logistic models adjusting for age and sex with "
    "family-clustered SEs (lenient case definition). KSADS onset = first met criteria (present, "
    "past, or partial remission) on the parent KSADS-COMP at Year 4 or Year 6 with documented "
    "absence at Baseline and Year 2. Anxiety = generalized anxiety, separation anxiety, social "
    "anxiety, panic, and agoraphobia; Externalizing = ADHD, oppositional defiant, or conduct "
    "disorder. The two panels differ only in the control pool. KSADS = Kiddie Schedule for "
    "Affective Disorders and Schizophrenia. * p < .05. ** p < .01. *** p < .001.",
    tsize=9.5, group_rows=gr, first_col_align=WD_ALIGN_PARAGRAPH.LEFT)

# ---- S6 child (youth-report) KSADS ----
# (cases, mesor, amp, acro) from 20_ksads_child_threecat_hc.log; lenient where defined.
# Youth conduct is not assessed at Baseline/Year 2, so externalizing lenient is undefined;
# the conservative estimate is shown for externalizing only.
ksc = {
    ("CBCL HC", "Depression"): (1199, "1.42 [1.29, 1.57]***", "0.80 [0.73, 0.89]***", "1.25 [1.13, 1.38]***"),
    ("CBCL HC", "Anxiety"):     (851, "1.40 [1.27, 1.55]***", "0.80 [0.73, 0.89]***", "1.09 [0.99, 1.21]"),
    ("CBCL HC", "Externalizing"): (310, "1.46 [1.25, 1.69]***", "0.99 [0.84, 1.15]", "1.42 [1.21, 1.66]***"),
    ("KSADS HC", "Depression"): (1199, "1.13 [1.05, 1.21]**", "0.93 [0.86, 1.00]*", "1.10 [1.03, 1.19]**"),
    ("KSADS HC", "Anxiety"):    (851, "1.12 [1.03, 1.21]**", "0.90 [0.83, 0.98]*", "0.97 [0.89, 1.06]"),
    ("KSADS HC", "Externalizing"): (310, "1.18 [1.05, 1.33]**", "1.03 [0.92, 1.17]", "1.24 [1.11, 1.40]***"),
}
s6_rows = []
s6_rows.append(("CBCL healthy controls (n = 1,188)", "", "", "", ""))
gr6 = {0}
for cat_ in ["Depression", "Anxiety", "Externalizing"]:
    c, m, a, ac = ksc[("CBCL HC", cat_)]
    s6_rows.append((cat_, str(c), m, a, ac))
s6_rows.append(("Youth-KSADS healthy controls (n = 3,464)", "", "", "", ""))
gr6.add(len(s6_rows) - 1)
for cat_ in ["Depression", "Anxiety", "Externalizing"]:
    c, m, a, ac = ksc[("KSADS HC", cat_)]
    s6_rows.append((cat_, str(c), m, a, ac))
add_table(
    doc, "Table S6",
    "Child-Report KSADS-COMP Diagnostic Onset Predicted by Wave-2 Cardiac Rhythm (Exploratory Convergent Check)",
    ["Category", "Cases", "Mesor", "Amplitude", "Acrophase"],
    s6_rows,
    "Cells are odds ratios [95% CI] per 1-SD; parallel to Table S5 but using the youth self-report "
    "KSADS-COMP (lenient case definition; age and sex adjusted, family-clustered SEs). The youth "
    "interview covers fewer modules: Anxiety = generalized anxiety, social anxiety, and panic; "
    "Externalizing = conduct disorder only (youth report has no ADHD or oppositional defiant "
    "disorder, and conduct is not assessed before Year 4, so the conservative definition is used). "
    "Youth self-report endorses threshold more often than parent report, so case counts are not "
    "comparable to Table S5. The two panels differ only in the control pool. "
    "* p < .05. ** p < .01. *** p < .001.",
    tsize=9.5, group_rows=gr6, first_col_align=WD_ALIGN_PARAGRAPH.LEFT)

OUT.mkdir(parents=True, exist_ok=True)
path = OUT / "Supplementary_Materials.docx"
doc.save(path)
print("Wrote", path)
