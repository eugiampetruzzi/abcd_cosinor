"""03 · Primary onset analyses — DSM Depression.

Runs the modeling framework (logistic regression with age + sex covariates and
family-clustered SEs) for each Wave-2 cosinor BLUP and each within-person SD
predictor, plus joint-with-mesor models for the within-person predictors.

Inputs:
    results/tables/analytic_depression.tsv         (from 01_…)
    derivatives/within_person_sensitivity/within_person_features.csv

Outputs:
    results/tables/primary_depression_results.tsv
    results/outputs/primary_depression_results.log
"""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.primary_runner import run_primary_analysis  # noqa: E402


def main() -> None:
    run_primary_analysis(
        slug="depression",
        outcome_label="DSM Depression",
        out_filename="primary_depression_results.tsv",
    )


if __name__ == "__main__":
    main()
