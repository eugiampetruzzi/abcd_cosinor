"""Figure 5 — Cosinor heart-rate rhythm in incident-hypertension cases vs.
normotensive controls. Thin wrapper around `utils.fig_builder`.
"""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.primary_runner import load_analytic_frame  # noqa: E402
from utils.fig_builder import build_combined_figure   # noqa: E402
from utils.paths import FIGS_DIR                      # noqa: E402


def main() -> None:
    df = load_analytic_frame("hypertension")
    print(f"Cohort n = {len(df):,}, cases = {int(df['onset'].sum())}")
    build_combined_figure(
        df,
        case_label="Incident hypertension",
        control_label="Normotensive controls",
        out_dir=FIGS_DIR,
        out_stem="fig5_hypertension",
    )


if __name__ == "__main__":
    main()
