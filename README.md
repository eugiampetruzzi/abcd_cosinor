# Cardiac rhythm development: A wearable device index of risk for physical and mental illness in adolescence

We characterize the 24-hour cardiac rhythm across adolescence from up to 21 days
of Fitbit wear at each of three biennial waves in the ABCD Study, and test
whether individual differences in the rhythm and its development forecast the
onset of psychopathology and cardiometabolic illness. Cosinor mixed-effects
models yield three rhythm parameters per session — **mesor** (24-hour mean),
**amplitude** (diurnal swing), and **acrophase** (peak timing).

## Repository layout (in paper order)

| Folder | Manuscript element | Contents |
|---|---|---|
| `utils/` | shared modeling/data layer | canonical paths, outcome-flag construction, incidence definition, logistic-with-clustered-SE + BH-FDR, primary-analysis runner |
| `00_cosinor_modeling/` | Methods → Cardiac rhythm modeling; **Figure 1** | cosinor mixed-effects fit (`cosinor_fit.R`), fit QC (`cosinor_qc.py`), and the Figure 1 schematic (`fig01_cosinor_schematic.py`) |
| `01_participants/` | Results → Participants; **Table 1** | participant-characteristics table (full vs. prediction sample) |
| `02_development/` | **Hypothesis 1: Development of the rhythm**; **Figure 2**, **Table 2** | age/sex GAMM trajectories, GAMLSS centiles, cross-wave stability, activity/sleep adjustment, pubertal-stage sex mediation, demographic associations |
| `03_codevelopment/` | **Hypothesis 2 — Co-development**; **Table S2** | bivariate parallel-process latent growth models (mesor × each outcome) for psychopathology and cardiometabolic trajectories |
| `04_clinical_onset/` | **Hypothesis 2 — Clinical onset**; **Tables S3–S4**, **Figure 3** | sample/incidence definition, hierarchical M1→M4 incremental models, transdiagnostic between-/within-person onset (FDR-30), and the two-panel rhythm forest |
| `05_ksads_convergent/` | **Exploratory convergent check**; **Tables S5–S6** | KSADS-COMP diagnostic onset, parent (S5) and youth (S6) report |
| `06_supplement/` | **Supplementary Materials** | builds the full supplement document (Tables S1–S6) |

## Data

All data were drawn from Release 6.1 of the Adolescent Brain Cognitive Development (ABCD) Study, a multi-site longitudinal cohort of U.S. adolescents recruited at
ages 9-10. All ABCD procedures were approved by a central IRB and site-level IRBs; parents provided written informed consent and youth provided written assent. 

## Definitions (shared across analyses)

- **Waves**: W1 = ses-00A (baseline), W2 = ses-02A (predictor wave), W3 = ses-04A, W4 = ses-06A.
- **Clinical threshold**: CBCL DSM-oriented T ≥ 65; obesity = BMI ≥ 85th CDC percentile; hypertension = SBP/DBP ≥ 95th percentile (<13 yr) or ≥ 130/80 mmHg (≥13 yr).
- **Incident onset**: first crossing of threshold at W3 or W4 with documented absence at all prior observed waves.
- **Ultra-clean healthy controls (n = 1,188)**: below clinical threshold on all six CBCL DSM scales and on obesity and hypertension at every observed wave.
- All onset models use per-1-SD z-scored predictors with age + sex and family-clustered SEs; FDR is Benjamini-Hochberg.

## Software

Python 3.12 (pandas, polars, numpy, statsmodels, scikit-learn, scipy,
python-docx, matplotlib) and R 4.5 (lme4, lavaan, mgcv, gamlss). See
`requirements.txt`.

Claude Code (Anthropic) was used to assist with code drafting and debugging. All code was reviewed, and verified by the authors, who checked outputs against expected results and confirmed the correctness of all analyses.
