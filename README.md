# Heart rate rhythm from a wearable predicts depression and cardiometabolic illness in adolescence

Code for *Heart rate rhythm from a wearable predicts depression and cardiometabolic illness in adolescence* (Giampetruzzi, Kircanski, & Gotlib, under review).

## Layout

```
code/
  utils/           shared loaders, modeling helpers (family-clustered logistic regression)
  analyses/        numbered pipeline scripts + build_supplement.py
  figures/         main-text figure scripts
results/
  tables/          analytic frames, primary results, sample/incidence summaries
  outputs/         per-script run logs
  sensitivity/     multicomponent, behavioral horserace, within-person sensitivity
figures/           main-text figures (PNG, PDF, SVG)
supplement/        supplement.docx + tables/ + figures/ (300+ DPI PNGs)
```

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Data

Data used in this study are available as part of the Adolescent Brain Cognitive Development (ABCD) Study, Release 6.0 (https://doi.org/10.82525/jy7n-g441) 

## Pipeline

| Script | Purpose |
|---|---|
| `01_sample_and_incidence.py` | CONSORT flow, per-wave outcome matrix, single HC group (N=2,004), analytic frames per outcome |
| `02_cosinor_descriptives.py` | BLUP and R^2 distributions |
| `03_primary_depression.py` | Primary onset models for CBCL DSM-Depression |
| `04_primary_obesity.py` | Primary onset models for BMI >= 85th percentile |
| `05_primary_hypertension.py` | Primary onset models for SBP/DBP >= 95th percentile |
| `06_sleep_activity_covariates.py` | Sleep + activity covariate adjustment |
| `07a_conditional_prediction_primary.py` | Cross-condition (comorbidity) prediction |
| `07b_conditional_prediction_sensitivity.py` | Wave-1-restricted sensitivity |
| `13_multicomponent_cosinor_sensitivity.py` | 24+12-hour cosinor sensitivity |
| `16_hc_vs_incident_demographics.py` | Demographics: HC vs each incident-case group |
| `17_incremental_predictive_value.py` | Incremental predictive value of cosinor over baseline covariates |
| `18_rhythm_horserace_nested.py` | 4-model nested behavioral adjustment (HR / +activity / +sleep / +both) |
| `18b_htn_amplitude_stability.py` | Amplitude stability sensitivity for hypertension |
| `18c_within_person_sd_matched_3signal.py` | Within-person SDs adjusted for scale-matched behavioral SDs |
| `19_table1_new_cohort.py` | Table 1 descriptives for the new analytic cohort |
| `build_supplement.py` | Assembles the full supplement Word document |

Run a script from the repo root: `python code/analyses/01_sample_and_incidence.py`.


