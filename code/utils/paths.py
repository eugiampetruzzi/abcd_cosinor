"""Canonical paths for the fitbit_prediction_github project.

Single source of truth for every analysis. Update here if anything moves on
disk. All other modules import only from this file.
"""
from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
ONEDRIVE = Path("/Users/eu/Library/CloudStorage/OneDrive-Stanford/"
                "Research Projects/1 - Data/ABCD")
PAPER    = ONEDRIVE / "ABCD Actigraphy Resource Paper"

# Existing dairc working tree (raw → derivatives pipeline)
DAIRC    = PAPER / "dairc"
DERIV    = DAIRC / "derivatives"          # per-subject derivatives (cosinor BLUPs, etc.)
QC       = PAPER / "qc"                   # CONSORT inputs, stage-1 cohort tables

# Outcomes (master per-wave longitudinal frames assembled previously)
OUTCOMES_DIR = PAPER / "outcomes"
MH_OUTCOMES   = OUTCOMES_DIR / "master_outcomes_mental_health.parquet"
PHYS_OUTCOMES = OUTCOMES_DIR / "master_outcomes_physical_health.parquet"

# Demographics (Release 6.1)
DEMO_DIR = ONEDRIVE / "Release 6.1" / "Demographics" / "phenotype"

# Wave-2 cosinor BLUPs (predictor: typical-day mesor / amplitude / acrophase)
COSINOR_BLUP_W2 = DERIV / "cosinor_features" / "per_wave" / "ses-02A" / "participant_blups.parquet"

# Wave-2 typical-day hourly profile (24 rows per subject)
HOURLY_PROFILE_W2 = DERIV / "hourly_profiles" / "ses-02A.parquet"

# Family structure for clustered SEs
FAMILY_PARQUET = DERIV / "family_structure.parquet"

# Within-person daily-cosinor features (built upstream by within_person_sensitivity.py)
WITHIN_PERSON_FEATURES = (DERIV / "within_person_sensitivity"
                                  / "within_person_features.csv")

# CDC BMI LMS table for percentile computation
CDC_BMI_LMS = DAIRC / "code" / "utils" / "cdc_bmi_lms.csv"

# Box-Box mount (life events; not used in primary onset)
BOX = Path("/Users/eu/Library/CloudStorage/Box-Box/mooddata_nophi/ABCD")

# ---------------------------------------------------------------------------
# Outputs (this project)
# ---------------------------------------------------------------------------
THIS = Path(__file__).resolve().parents[2]   # fitbit_prediction_github/
RESULTS_DIR = THIS / "results"
TABLES_DIR  = RESULTS_DIR / "tables"
OUTPUTS_DIR = RESULTS_DIR / "outputs"
FIGS_DIR    = THIS / "figures"

# ---------------------------------------------------------------------------
# Wave name conventions used across all scripts
# ---------------------------------------------------------------------------
W1 = "ses-00A"   # baseline (~9-10 yr)
W2 = "ses-02A"   # 2-year FU (~11-12 yr)  — Fitbit predictor wave
W3 = "ses-04A"   # 4-year FU (~13-14 yr)  — first outcome wave
W4 = "ses-06A"   # 6-year FU (~15-16 yr)  — second outcome wave

PRE_WAVES_FULL = (W1, W2)   # mental health + obesity have W1 measurement
PRE_WAVES_BP   = (W2,)      # blood pressure not measured at baseline
ONSET_WAVES    = (W3, W4)
