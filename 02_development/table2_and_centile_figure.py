"""Build cleaner Figure 3 (shaded-band growth charts) + supplementary
reference table of centile values at ages 10, 12, 14, 16, 18 for
mesor / amplitude / acrophase x male / female.

Outputs:
    fig3_centile_growth_charts.png  / .pdf
    Table_Centile_Reference.docx
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = Path(os.path.dirname(os.path.abspath(__file__)))
CENTILES = pd.read_csv(
    "/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/"
    "ABCD/ABCD Actigraphy Resource Paper/dairc/derivatives/"
    "paper2_normative_centiles/centiles_all.tsv"
)

PARAMS = [("MESOR",     "Mesor (bpm)"),
          ("Amplitude", "Amplitude (bpm)"),
          ("Acrophase", "Acrophase (clock hour)")]
SEXES  = [("male",   "Males",   "#2C7FB8"),
          ("female", "Females", "#E66101")]

# Pivot to wide: rows = (param, sex, age), cols = centiles
wide = (CENTILES.pivot_table(index=["param","sex","age_yrs"],
                             columns="centile", values="value")
                .reset_index())

# Figure 3: shaded-band growth charts, 2 rows (sex) x 3 cols (parameter)
fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)

BAND_PAIRS = [(3, 97, 0.12, "3rd–97th"),
              (10, 90, 0.22, "10th–90th"),
              (25, 75, 0.34, "25th–75th")]

for r, (sex, sex_lab, color) in enumerate(SEXES):
    for c, (param, plab) in enumerate(PARAMS):
        ax = axes[r, c]
        sub = wide[(wide["param"] == param) & (wide["sex"] == sex)].sort_values("age_yrs")
        x = sub["age_yrs"].values
        for lo, hi, alpha, _ in BAND_PAIRS:
            ax.fill_between(x, sub[lo].values, sub[hi].values,
                            color=color, alpha=alpha, linewidth=0)
        ax.plot(x, sub[50].values, color=color, linewidth=1.8)
        ax.set_xlim(10, 18); ax.set_xticks([10, 12, 14, 16, 18])
        if r == 1:
            ax.set_xlabel("Age (years)", fontsize=10)
        if c == 0:
            ax.set_ylabel(sex_lab, fontsize=11, fontweight="bold")
        if r == 0:
            ax.set_title(plab, fontsize=11)
        ax.tick_params(axis="both", labelsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

# Single legend in last panel
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="gray", alpha=0.34, label="25th–75th centile"),
    Patch(facecolor="gray", alpha=0.22, label="10th–90th centile"),
    Patch(facecolor="gray", alpha=0.12, label="3rd–97th centile"),
    plt.Line2D([0], [0], color="gray", linewidth=1.8, label="50th centile (median)"),
]
axes[0, 2].legend(handles=legend_handles, loc="lower left",
                  fontsize=8, frameon=False, bbox_to_anchor=(1.02, 0.05))

fig.tight_layout(w_pad=1.5, h_pad=1.2)
png = OUT / "fig3_centile_growth_charts.png"
pdf = OUT / "fig3_centile_growth_charts.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure: {png}")

# Supplementary reference table
REF_AGES = [10, 12, 14, 16, 18]
CENT_COLS = [3, 10, 25, 50, 75, 90, 97]
HEAD = ["3rd", "10th", "25th", "50th", "75th", "90th", "97th"]

# Pull values nearest to integer ages
ref = (CENTILES[CENTILES["age_yrs"].isin([float(a) for a in REF_AGES])]
       .pivot_table(index=["param","sex","age_yrs"], columns="centile",
                    values="value")
       .reset_index())


def fmt_val(v, param):
    return f"{v:.1f}" if param != "Acrophase" else f"{v:.2f}"


doc = Document()
style = doc.styles["Normal"]
style.font.name = "Arial"; style.font.size = Pt(12)
style.paragraph_format.line_spacing = 2.0
style.paragraph_format.space_after = Pt(0)
style.paragraph_format.space_before = Pt(0)
for s in doc.sections:
    s.top_margin = Inches(1); s.bottom_margin = Inches(1)
    s.left_margin = Inches(1); s.right_margin = Inches(1)


def set_cell(cell, text, bold=False, italic=False, align="center", size=11):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = (WD_ALIGN_PARAGRAPH.CENTER if align == "center"
                   else WD_ALIGN_PARAGRAPH.LEFT)
    run = p.add_run(text)
    run.font.name = "Arial"; run.font.size = Pt(size)
    run.bold = bold; run.italic = italic
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15


def _border(cell, sides_on):
    tc = cell._tc; tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for side in ["top","left","bottom","right"]:
        el = OxmlElement(f"w:{side}")
        if side in sides_on:
            el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "4"); el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "none"); el.set(qn("w:sz"), "0")
        tcBorders.append(el)
    tcPr.append(tcBorders)


def no_border(cell): _border(cell, [])
def border_top_bottom(cell): _border(cell, ["top", "bottom"])
def border_bottom(cell): _border(cell, ["bottom"])


def add_p(text, bold=False, italic=False, size=12, space_after=0):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Arial"; run.font.size = Pt(size)
    run.bold = bold; run.italic = italic
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    return p


add_p("Table S1", bold=True, space_after=0)
add_p("Sex-Specific Centile Reference Values for 24-Hour Cardiac Rhythm Parameters at Ages 10, 12, 14, 16, and 18 Years",
      italic=True, space_after=6)

ncols = 2 + len(CENT_COLS)  # Age, Sex, 7 centiles
nrows_header = 2
nrows_body = sum(len(SEXES) * len(REF_AGES) for _ in PARAMS) + len(PARAMS)  # +parameter section header rows
table = doc.add_table(rows=nrows_header + nrows_body, cols=ncols)

# Width fix
widths = [Inches(0.6), Inches(0.75)] + [Inches(0.75)] * len(CENT_COLS)
for i, w in enumerate(widths):
    for row in table.rows:
        row.cells[i].width = w

# Row 0: top heading
set_cell(table.cell(0, 0), "Age (yr)", align="center", size=11)
set_cell(table.cell(0, 1), "Sex", align="center", size=11)
for j, h in enumerate(HEAD):
    set_cell(table.cell(0, 2 + j), h + " centile", align="center", size=11)
for j in range(ncols):
    border_top_bottom(table.cell(0, j))

# Row 1: units block  used for parameter labels embedded in groups; leave blank
for j in range(ncols):
    set_cell(table.cell(1, j), "", align="center", size=11)
    no_border(table.cell(1, j))

# Body
r = 1
for pi, (param, plab) in enumerate(PARAMS):
    r += 1
    # Section header row spanning full width: parameter name
    sect_cell = table.cell(r, 0)
    # Merge across cols
    for j in range(1, ncols):
        sect_cell = sect_cell.merge(table.cell(r, j))
    set_cell(sect_cell, plab, bold=False, italic=True, align="left", size=11)
    border_bottom(sect_cell)

    for sex, sex_lab, _ in SEXES:
        for age in REF_AGES:
            r += 1
            row = ref[(ref["param"] == param) & (ref["sex"] == sex) &
                      (ref["age_yrs"] == float(age))]
            set_cell(table.cell(r, 0), str(age), align="center", size=11)
            set_cell(table.cell(r, 1), sex_lab[:-1], align="center", size=11)
            if len(row) == 0:
                for j, _ in enumerate(CENT_COLS):
                    set_cell(table.cell(r, 2 + j), "—", align="center", size=11)
            else:
                row = row.iloc[0]
                for j, c in enumerate(CENT_COLS):
                    set_cell(table.cell(r, 2 + j),
                             fmt_val(row[c], param), align="center", size=11)
            for j in range(ncols):
                no_border(table.cell(r, j))

# Bottom border on the very last row
for j in range(ncols):
    _border(table.cell(r, j), ["bottom"])

add_p("")
note = doc.add_paragraph()
note.paragraph_format.line_spacing = 1.15
n1 = note.add_run("Note. "); n1.font.name = "Arial"; n1.font.size = Pt(11); n1.italic = True
n2 = note.add_run(
    "Values are sex-specific centiles of each Wave-2 cardiac rhythm parameter "
    "at integer ages, estimated from GAMLSS models with Box–Cox t (BCT) "
    "distributions and smoothing splines for the location, scale, and shape "
    "parameters. Each centile represents the value exceeded by the indicated "
    "percentage of same-sex peers at that age (e.g., a male at age 14 with a "
    "mesor at the 75th centile has a higher 24-h mean heart rate than 75% of "
    "same-aged male peers). Mesor and amplitude are reported in beats per "
    "minute; acrophase is reported as clock hour of peak heart rate. "
    "N = 8,301 adolescents contributing 13,552 sessions across Waves 2, 4, "
    "and 6. GAMLSS = generalized additive models for location, scale and "
    "shape; bpm = beats per minute."
)
n2.font.name = "Arial"; n2.font.size = Pt(11)

out_path = OUT / "Table_Centile_Reference.docx"
doc.save(out_path)
print(f"Saved table: {out_path}")
