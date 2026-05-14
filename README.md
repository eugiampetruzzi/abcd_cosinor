# abcd_cosinor

Wearable heart-rate rhythm as a prospective predictor of depression, obesity, and hypertension in adolescence.

Companion code for the manuscript *Wearable Heart Rate Rhythm Predicts Depression and Cardiometabolic Illness in Adolescence*, using Wave-2 Fitbit data from the ABCD Novel Technologies sub-study to fit per-participant single-component cosinor models and test whether typical-day rhythm features (mesor, amplitude, acrophase) and within-person daily-cosinor SDs predict incident clinical thresholds at Waves 3 and 4.

## Repo layout

```
code/
  utils/           shared loaders, modeling helpers (family-clustered logistic regression)
  analyses/        numbered pipeline scripts (01-15) + build_supplement.py
  figures/         main-text figure scripts
results/
  tables/          analytic frames, primary results, sample/incidence summaries
  outputs/         per-script run logs
  sensitivity/     multicomponent, superhealthy, within-person sensitivity outputs
figures/           main-text figures (PNG, PDF, SVG)
supplement/        supplement.docx + tables/ + figures/ (300+ DPI PNGs)
```

## Pipeline order

1. `01_sample_and_incidence.py` — CONSORT flow, per-wave outcome matrix, common-HC and incident-case definitions, analytic frames per outcome
2. `02_cosinor_descriptives.py` — BLUP and R^2 distributions
3. `03_primary_depression.py`, `04_primary_obesity.py`, `05_primary_hypertension.py` — primary onset models
4. `06_sleep_activity_covariates.py` — sleep/activity covariate adjustment
5. `07a_conditional_prediction_primary.py`, `07b_conditional_prediction_sensitivity.py` — cross-condition prediction
6. `13_multicomponent_cosinor_sensitivity.py` — 24+12-hour cosinor sensitivity
7. `14_superhealthy_sensitivity.py`, `15_superhealthy_demographics.py` — super-healthy HC sensitivity + demographics
8. `build_supplement.py` — assembles the supplement Word document

Run a script from the repo root: `python -m code.analyses.01_sample_and_incidence`, or directly: `python code/analyses/01_sample_and_incidence.py`.

## Conventions

- Predictors are per-1-SD z-scored within each analytic frame.
- All onset models are logistic regression with age and sex fixed effects and family-clustered standard errors on ABCD family ID (`utils.modeling.fit_logistic_cluster`).
- Outcome flags: `dsm_dep_65` (CBCL DSM-Depression T >= 65), `obese_85` (CDC BMI percentile >= 85), `htn` (SBP >= 130 OR DBP >= 80).
- Incident case: first observed elevation at Wave 3 or Wave 4 with documented absence of elevation at all earlier observed assessments.
- Single HC group (N = 2,004): below threshold for depression, obesity, AND hypertension at every observed wave with confirmed below-threshold follow-up CBCL, BMI, and BP at W3 or W4.

## Data dependencies

ABCD raw data is access-restricted and not redistributed here. The pipeline expects:

```
~/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/ABCD/
  Release 6.1/Demographics/phenotype/ab_g_stc.tsv
  ABCD Actigraphy Resource Paper/
    outcomes/master_outcomes_{mental_health,physical_health}.parquet
    qc/stage1_full_cohort_sessions.tsv
    qc/stage3_validation/mesor_vs_clinic_hr.tsv
    dairc/derivatives/cosinor_features/per_wave/ses-02A/participant_blups.parquet
    dairc/derivatives/cosinor_features/pooled/participant_blups.parquet
    dairc/derivatives/family_structure.parquet
    dairc/derivatives/hourly_profiles/ses-02A.parquet
    dairc/derivatives/within_person_sensitivity/within_person_features.csv
    dairc/derivatives/fitbit_summary/per_wave_summary.parquet
```

Adjust `code/utils/paths.py` if your local paths differ.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Citation

If you use this code, please cite the manuscript (DOI added on acceptance).

## License

MIT.
