library(lavaan)
library(arrow)
library(dplyr)
library(tidyr)
library(readr)

ONEDRIVE <- file.path("/Users/eu/Library/CloudStorage/OneDrive-Stanford",
                       "Research Projects/1 - Data/ABCD")
PAPER   <- file.path(ONEDRIVE, "ABCD Actigraphy Resource Paper")
DERIV   <- file.path(PAPER, "dairc/derivatives")

blups_all <- bind_rows(
  read_parquet(file.path(DERIV, "cosinor_features/per_wave/ses-02A/participant_blups.parquet")) |>
    mutate(session_id = "ses-02A"),
  read_parquet(file.path(DERIV, "cosinor_features/per_wave/ses-04A/participant_blups.parquet")) |>
    mutate(session_id = "ses-04A"),
  read_parquet(file.path(DERIV, "cosinor_features/per_wave/ses-06A/participant_blups.parquet")) |>
    mutate(session_id = "ses-06A")
) |> rename(participant_id = subject_id) |> filter(!is.na(r_squared))

mesor_wide <- blups_all |>
  select(participant_id, session_id, mesor_blup) |>
  pivot_wider(id_cols = participant_id, names_from = session_id,
              values_from = mesor_blup, names_prefix = "mesor_")

phys <- read_parquet(file.path(PAPER, "outcomes/master_outcomes_physical_health.parquet"))

demo <- read_tsv(
  file.path(ONEDRIVE, "Release 6.1/Actigraphy_Eu_Outputs/subject_demographics.tsv"),
  show_col_types = FALSE
)

pp_syn <- '
  i_m =~ 1*m2 + 1*m4 + 1*m6
  s_m =~ 0*m2 + 1*m4 + 2*m6
  i_s =~ 1*s2 + 1*s4 + 1*s6
  s_s =~ 0*s2 + 1*s4 + 2*s6
  s_m ~~ s_s
  i_m ~~ i_s
  s_s ~ i_m
  i_m ~ is_female
  s_m ~ is_female
  i_s ~ is_female
  s_s ~ is_female
'

run_pp <- function(outcome_wide, label) {
  cat(sprintf("\n========== %s ==========\n", label))

  pp_df <- mesor_wide |>
    inner_join(outcome_wide, by = "participant_id") |>
    left_join(demo |> select(participant_id, sex, site_baseline), by = "participant_id") |>
    mutate(is_female = as.numeric(sex == 2)) |>
    rename(m2 = `mesor_ses-02A`, m4 = `mesor_ses-04A`, m6 = `mesor_ses-06A`)

  n_both <- sum(rowSums(!is.na(pp_df[c("m2","m4","m6")])) >= 2 &
                rowSums(!is.na(pp_df[c("s2","s4","s6")])) >= 2)
  cat("  N with 2+ waves both:", n_both, "\n")

  fit <- growth(pp_syn, data = pp_df, missing = "fiml", estimator = "MLR",
                cluster = "site_baseline")
  est <- parameterEstimates(fit, standardized = TRUE)
  fm <- fitMeasures(fit, c("cfi", "rmsea"))

  ss <- est |> filter(op == "~~", lhs == "s_m", rhs == "s_s")
  ii <- est |> filter(op == "~~", lhs == "i_m", rhs == "i_s")
  cr <- est |> filter(op == "~", lhs == "s_s", rhs == "i_m")

  cat(sprintf("  N = %d, CFI = %.3f, RMSEA = %.3f\n",
    lavInspect(fit, "nobs"), fm["cfi"], fm["rmsea"]))
  cat(sprintf("  Slope-slope r = %.3f, p = %s\n",
    ss$std.all[1], formatC(ss$pvalue[1], format="g", digits=3)))
  cat(sprintf("  Intercept-intercept r = %.3f, p = %s\n",
    ii$std.all[1], formatC(ii$pvalue[1], format="g", digits=3)))
  cat(sprintf("  Cross (mesor int -> %s slope): beta = %.3f, p = %s\n",
    label, cr$std.all[1], formatC(cr$pvalue[1], format="g", digits=3)))
}

# BMI
bmi_wide <- phys |>
  filter(session_id %in% c("ses-02A", "ses-04A", "ses-06A")) |>
  select(participant_id, session_id, bmi) |>
  filter(!is.na(bmi)) |>
  pivot_wider(id_cols = participant_id, names_from = session_id,
              values_from = bmi, names_prefix = "bmi_") |>
  rename(s2 = `bmi_ses-02A`, s4 = `bmi_ses-04A`, s6 = `bmi_ses-06A`)

run_pp(bmi_wide, "BMI")

# Systolic BP
bp_wide <- phys |>
  filter(session_id %in% c("ses-02A", "ses-04A", "ses-06A")) |>
  select(participant_id, session_id, bp_sys_mean) |>
  filter(!is.na(bp_sys_mean)) |>
  pivot_wider(id_cols = participant_id, names_from = session_id,
              values_from = bp_sys_mean, names_prefix = "bp_") |>
  rename(s2 = `bp_ses-02A`, s4 = `bp_ses-04A`, s6 = `bp_ses-06A`)

run_pp(bp_wide, "Systolic BP")

cat("\nDone.\n")
