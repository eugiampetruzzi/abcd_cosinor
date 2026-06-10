# abcd_cosinor

Analysis code for **"Cardiac rhythm development: A wearable device index of risk
for physical and mental illness in adolescence"** (Giampetruzzi, Kircanski,
Pine, & Gotlib).

We characterize the 24-hour cardiac rhythm across adolescence from up to 21 days
of Fitbit wear at each of three biennial waves in the ABCD Study, and test
whether individual differences in the rhythm and its development forecast the
onset of psychopathology and cardiometabolic illness. Cosinor mixed-effects
models yield three rhythm parameters per session — **mesor** (24-hour mean),
**amplitude** (diurnal swing), and **acrophase** (peak timing).

The directory layout follows the order of the manuscript.

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

## Element → script map

**Figure 1** (cosinor schematic) — `00_cosinor_modeling/fig01_cosinor_schematic.py`
**Table 1** (participants) — `01_participants/table1_participants.py`
**Figure 2** (development trajectories) — `02_development/a_rhythm_age_trajectories.py`, `fig02_development_trajectories.py`
**Table 2** (centile reference values) — `02_development/c_centiles_gamlss.R` → `table2_and_centile_figure.py`
**Co-development (Table S2)** — `03_codevelopment/parallel_process_psychopathology.R` (psychopathology) + `parallel_process_cardiometabolic.R` (BMI, systolic BP)
**Clinical onset (Tables S3–S4, Figure 3)** — `04_clinical_onset/three_category_onset.py` is the primary driver (3 psychopathology categories + obesity + hypertension vs. the n = 1,188 ultra-clean control pool); `ultraclean_full_rerun.py` and `incremental_all_dsm.py` reproduce the full 8-outcome version; `fig03_rhythm_forest.py` builds Figure 3.
**KSADS convergent (Tables S5–S6)** — `05_ksads_convergent/parent_ksads_onset.py` (S5) and `child_ksads_onset.py` (S6), both importing `ksads_models_helpers.py`.
**Supplement (Tables S1–S6)** — `06_supplement/build_supplement.py`.

## Data

This repository contains **code only**. It depends on the ABCD Study 7.0 / 6.1
data releases (NDA), which are access-restricted and not redistributable, plus
derived cosinor BLUPs and per-wave Fitbit summaries produced by
`00_cosinor_modeling/`. All absolute file locations are centralized in
`utils/paths.py`; set them to your local ABCD mirror before running. Scripts use
`sys.path` insertion to import `utils`; adjust the inserted path to this
repository root for your environment.

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

Code was developed with the assistance of Claude Code (Anthropic).
