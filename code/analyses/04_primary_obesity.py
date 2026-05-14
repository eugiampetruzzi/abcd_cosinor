"""04 · Primary onset analyses — Overweight/Obesity.

Same modeling framework as `03_primary_depression.py`. See `utils/primary_runner.py`.

Outputs:
    results/tables/primary_obesity_results.tsv
    results/outputs/primary_obesity_results.log
"""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.primary_runner import run_primary_analysis  # noqa: E402


def main() -> None:
    run_primary_analysis(
        slug="obesity",
        outcome_label="Overweight/Obesity",
        out_filename="primary_obesity_results.tsv",
    )


if __name__ == "__main__":
    main()
