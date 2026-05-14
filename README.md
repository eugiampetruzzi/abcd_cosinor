# abcd_cosinor

Companion code for *Heart rate rhythm from a wearable predicts depression and cardiometabolic illness in adolescence* (Giampetruzzi, Kircanski, & Gotlib).

Wave-2 Fitbit data from the ABCD Novel Technologies sub-study are used to fit per-participant single-component cosinor models. Three rhythm parameters (mesor, amplitude, acrophase) and three within-person stability indices (SD of daily mesor, amplitude, acrophase) are tested as prospective predictors of incident depression, obesity, and hypertension at Waves 3-4.

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

## Conventions

- Predictors are per-1-SD z-scored within each analytic frame.
- Logistic onset models include age and sex fixed effects and family-clustered SEs on ABCD family ID (`utils.modeling.fit_logistic_cluster`).
- Outcome flags: `dsm_dep_65` (CBCL DSM-Depression T >= 65), `obese_85` (CDC BMI percentile >= 85), `htn` (SBP/DBP >= 95th percentile or >=130/80 for ages >= 13).
- Incident case: first observed elevation at Wave 3 or Wave 4 with documented absence of elevation at every earlier observed wave.
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
