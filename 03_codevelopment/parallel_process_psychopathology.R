# 23 - Trajectory prediction across all CBCL DSM scales + parallel process
# Part A: Univariate LGC for each DSM scale (W2/W4/W6), with W2 cardiac
# rhythm predicting the latent slope (replicating script 20 structure).
# Hierarchical: M0 unconditional -> M1 demographics -> M2 baseline
# symptoms -> M3 behavioral -> M4 rhythm.
# Part B: Bivariate parallel-process LGC  mesor trajectory and symptom
# trajectory modeled jointly. Tests whether rhythm change covaries
# with symptom change (correlated slopes).
# Outputs:
# results/trajectory/dsm_all_lgc_summary.csv
# results/trajectory/dsm_all_m4_rhythm_effects.csv
# results/trajectory/parallel_process_summary.csv
# results/outputs/23_trajectory_all_dsm_and_parallel.log

library(lavaan)
library(arrow)
library(dplyr)
library(tidyr)
library(readr)

ONEDRIVE <- file.path(
  "/Users/eu/Library/CloudStorage/OneDrive-Stanford",
  "Research Projects/1 - Data/ABCD"
)
PAPER   <- file.path(ONEDRIVE, "ABCD Actigraphy Resource Paper")
DERIV   <- file.path(PAPER, "dairc/derivatives")
THIS    <- file.path(PAPER, "fitbit_prediction_superhealthy")
RESULTS <- file.path(THIS, "results")
TRAJ_DIR <- file.path(RESULTS, "trajectory")
dir.create(TRAJ_DIR, recursive = TRUE, showWarnings = FALSE)
OUT_DIR <- file.path(RESULTS, "outputs")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

log_lines <- character()
log_msg <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  log_lines <<- c(log_lines, msg)
}

# Load data

log_msg("Loading data...")

blups_w2 <- read_parquet(
  file.path(DERIV, "cosinor_features/per_wave/ses-02A/participant_blups.parquet")
) |> rename(participant_id = subject_id) |>
  select(participant_id, mesor_blup, amplitude_blup, acrophase_blup)

# All-wave BLUPs for parallel process models
blups_all <- bind_rows(
  read_parquet(file.path(DERIV, "cosinor_features/per_wave/ses-02A/participant_blups.parquet")) |>
    mutate(session_id = "ses-02A"),
  read_parquet(file.path(DERIV, "cosinor_features/per_wave/ses-04A/participant_blups.parquet")) |>
    mutate(session_id = "ses-04A"),
  read_parquet(file.path(DERIV, "cosinor_features/per_wave/ses-06A/participant_blups.parquet")) |>
    mutate(session_id = "ses-06A")
) |> rename(participant_id = subject_id) |>
  filter(!is.na(r_squared))

mh <- read_parquet(file.path(PAPER, "outcomes/master_outcomes_mental_health.parquet"))

demo <- read_tsv(
  file.path(ONEDRIVE, "Release 6.1/Actigraphy_Eu_Outputs/subject_demographics.tsv"),
  show_col_types = FALSE
)

beh <- read_parquet(
  file.path(DERIV, "fitbit_summary/per_wave_summary.parquet")
) |>
  filter(session_id == "ses-02A") |>
  select(participant_id, sleep_period_min, mvpa_min) |>
  filter(!is.na(sleep_period_min), !is.na(mvpa_min))

fam <- read_parquet(file.path(DERIV, "family_structure.parquet"))

# DSM scale definitions

scales <- list(
  dep   = list(col = "cbcl_dsm_dep_tscore",   label = "Depression"),
  anx   = list(col = "cbcl_dsm_anx_tscore",   label = "Anxiety"),
  adhd  = list(col = "cbcl_dsm_adhd_tscore",  label = "ADHD"),
  cond  = list(col = "cbcl_dsm_cond_tscore",  label = "Conduct"),
  opp   = list(col = "cbcl_dsm_opp_tscore",   label = "ODD"),
  somat = list(col = "cbcl_dsm_somat_tscore", label = "Somatic")
)

waves <- c("ses-00A", "ses-02A", "ses-04A", "ses-06A")
lgc_waves <- c("ses-02A", "ses-04A", "ses-06A")

# Helper functions

z_scale <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)

demo_vars <- c("is_female",
               "race_hispanic", "race_black", "race_asian", "race_other",
               "income_lt50k", "income_50_100k",
               "edu_lt_hs", "edu_hs", "edu_some_col", "edu_bachelors")

build_lgc_syntax <- function(covariates = NULL) {
  syn <- '
    intercept =~ 1*y2 + 1*y4 + 1*y6
    slope     =~ 0*y2 + 1*y4 + 2*y6
  '
  if (!is.null(covariates) && length(covariates) > 0) {
    cov_str <- paste(covariates, collapse = " + ")
    syn <- paste0(syn,
      "    intercept ~ ", cov_str, "\n",
      "    slope     ~ ", cov_str, "\n"
    )
  }
  syn
}

extract_fit <- function(fit, model_name) {
  fm <- fitMeasures(fit, c("cfi", "rmsea", "logl", "npar"))
  r2 <- tryCatch(lavInspect(fit, "r-square"), error = function(e) c())
  tibble(
    model = model_name,
    n = lavInspect(fit, "nobs"),
    cfi = as.numeric(fm["cfi"]),
    rmsea = as.numeric(fm["rmsea"]),
    r2_slope = if ("slope" %in% names(r2)) r2["slope"] else NA_real_
  )
}

# PART A: Univariate LGC for each DSM scale

log_msg("\n", strrep("=", 78))
log_msg("PART A: Univariate LGC — rhythm predicting symptom trajectories")
log_msg(strrep("=", 78))

all_lgc_results <- list()
all_m4_effects  <- list()

for (slug in names(scales)) {
  sc <- scales[[slug]]
  log_msg("\n", strrep("-", 60))
  log_msg(sc$label, " (", sc$col, ")")
  log_msg(strrep("-", 60))

  # Pivot CBCL to wide
  cbcl_wide <- mh |>
    filter(session_id %in% waves) |>
    select(participant_id, session_id, all_of(sc$col)) |>
    filter(!is.na(.data[[sc$col]])) |>
    pivot_wider(id_cols = participant_id, names_from = session_id,
                values_from = all_of(sc$col), names_prefix = "sc_")

  # Merge
  df <- blups_w2 |>
    inner_join(cbcl_wide, by = "participant_id") |>
    left_join(demo, by = "participant_id") |>
    left_join(beh, by = "participant_id") |>
    left_join(fam, by = "participant_id")

  w0_col <- "sc_ses-00A"
  df <- df |>
    rename(y2 = `sc_ses-02A`, y4 = `sc_ses-04A`, y6 = `sc_ses-06A`)
  if (w0_col %in% names(df)) {
    df <- df |> rename(y0 = `sc_ses-00A`)
  } else {
    df$y0 <- NA_real_
  }

  df$is_female <- as.numeric(df$sex == 2)
  df$race_hispanic <- as.numeric(df$ethnrace_label == "Hispanic")
  df$race_black    <- as.numeric(df$ethnrace_label == "Black")
  df$race_asian    <- as.numeric(df$ethnrace_label == "Asian")
  df$race_other    <- as.numeric(df$ethnrace_label == "Other")
  df$income_lt50k    <- as.numeric(df$income_3lvl_label == "<50k")
  df$income_50_100k  <- as.numeric(grepl("50.100k", df$income_3lvl_label))
  df$edu_lt_hs       <- as.numeric(df$edu_cgs_label == "<HS")
  df$edu_hs          <- as.numeric(df$edu_cgs_label == "HS/GED")
  df$edu_some_col    <- as.numeric(df$edu_cgs_label == "Some college")
  df$edu_bachelors   <- as.numeric(df$edu_cgs_label == "Bachelor's")

  df$mesor_z     <- z_scale(df$mesor_blup)
  df$amplitude_z <- z_scale(df$amplitude_blup)
  df$acrophase_z <- z_scale(df$acrophase_blup)
  df$sleep_z     <- z_scale(df$sleep_period_min)
  df$mvpa_z      <- z_scale(df$mvpa_min)
  df$baseline_z  <- z_scale(df$y0)

  n_2plus <- sum(rowSums(!is.na(df[c("y2", "y4", "y6")])) >= 2)
  log_msg("  N with cosinor + CBCL: ", nrow(df), " (2+ waves: ", n_2plus, ")")

  baseline_vars <- "baseline_z"
  behavioral_vars <- c("sleep_z", "mvpa_z")
  rhythm_vars <- c("mesor_z", "amplitude_z", "acrophase_z")

  # M0
  fit_m0 <- tryCatch(
    growth(build_lgc_syntax(), data = df, missing = "fiml",
           estimator = "MLR", cluster = "site_baseline"),
    error = function(e) { log_msg("  M0 failed: ", e$message); NULL }
  )
  if (is.null(fit_m0)) next

  m0_est <- parameterEstimates(fit_m0)
  m0_means <- m0_est |> filter(op == "~1", lhs %in% c("intercept", "slope"))
  int_mean <- m0_means$est[m0_means$lhs == "intercept"]
  slope_mean <- m0_means$est[m0_means$lhs == "slope"]
  slope_p <- m0_means$pvalue[m0_means$lhs == "slope"]
  log_msg(sprintf("  M0: intercept = %.2f, slope = %.3f (p = %s), n = %d",
    int_mean, slope_mean, formatC(slope_p, format = "g", digits = 3),
    lavInspect(fit_m0, "nobs")))

  # M1-M4
  fit_m1 <- growth(build_lgc_syntax(demo_vars), data = df,
                   missing = "fiml", estimator = "MLR", cluster = "site_baseline")
  fit_m2 <- growth(build_lgc_syntax(c(demo_vars, baseline_vars)), data = df,
                   missing = "fiml", estimator = "MLR", cluster = "site_baseline")
  fit_m3 <- growth(build_lgc_syntax(c(demo_vars, baseline_vars, behavioral_vars)),
                   data = df, missing = "fiml", estimator = "MLR",
                   cluster = "site_baseline")
  fit_m4 <- growth(build_lgc_syntax(c(demo_vars, baseline_vars, behavioral_vars,
                                      rhythm_vars)),
                   data = df, missing = "fiml", estimator = "MLR",
                   cluster = "site_baseline")

  s0 <- extract_fit(fit_m0, "M0"); s1 <- extract_fit(fit_m1, "M1")
  s2 <- extract_fit(fit_m2, "M2"); s3 <- extract_fit(fit_m3, "M3")
  s4 <- extract_fit(fit_m4, "M4")

  log_msg(sprintf("  M1 R2(slope)=%.4f  M2=%.4f  M3=%.4f  M4=%.4f  dR2(M3->M4)=%.4f",
    s1$r2_slope, s2$r2_slope, s3$r2_slope, s4$r2_slope,
    s4$r2_slope - s3$r2_slope))

  # LRT M3 vs M4
  lrt_p <- tryCatch({
    lrt <- lavTestLRT(fit_m3, fit_m4)
    lrt$`Pr(>Chisq)`[2]
  }, error = function(e) NA_real_)
  log_msg(sprintf("  LRT M3 vs M4: p = %s",
    if (is.na(lrt_p)) "failed" else formatC(lrt_p, format = "g", digits = 3)))

  # M4 rhythm effects on slope and intercept
  m4_est <- parameterEstimates(fit_m4, standardized = TRUE)
  for (target in c("slope", "intercept")) {
    rhy_eff <- m4_est |> filter(op == "~", lhs == target, rhs %in% rhythm_vars)
    for (i in seq_len(nrow(rhy_eff))) {
      r <- rhy_eff[i, ]
      log_msg(sprintf("    %s ~ %-14s  b = %7.4f, beta = %.3f, p = %s",
        target, r$rhs, r$est, r$std.all,
        formatC(r$pvalue, format = "g", digits = 3)))
      all_m4_effects[[length(all_m4_effects) + 1]] <- tibble(
        scale = sc$label, target = target, predictor = r$rhs,
        b = r$est, se = r$se, beta = r$std.all, p = r$pvalue,
        n = s4$n
      )
    }
  }

  all_lgc_results[[slug]] <- tibble(
    scale = sc$label,
    n_m0 = s0$n, intercept_mean = int_mean, slope_mean = slope_mean,
    slope_p = slope_p,
    r2_slope_m1 = s1$r2_slope, r2_slope_m2 = s2$r2_slope,
    r2_slope_m3 = s3$r2_slope, r2_slope_m4 = s4$r2_slope,
    delta_r2_rhythm = s4$r2_slope - s3$r2_slope,
    lrt_m3_m4_p = lrt_p,
    cfi_m4 = s4$cfi, rmsea_m4 = s4$rmsea
  )
}

lgc_summary <- bind_rows(all_lgc_results)
m4_effects  <- bind_rows(all_m4_effects)
write_csv(lgc_summary, file.path(TRAJ_DIR, "dsm_all_lgc_summary.csv"))
write_csv(m4_effects,  file.path(TRAJ_DIR, "dsm_all_m4_rhythm_effects.csv"))

log_msg("\n\nSummary — rhythm block (M4) effects on slope:")
for (sc_label in unique(m4_effects$scale)) {
  sub <- m4_effects |> filter(scale == sc_label, target == "slope")
  log_msg(sprintf("\n  %s (n = %d):", sc_label, sub$n[1]))
  for (i in seq_len(nrow(sub))) {
    r <- sub[i, ]
    sig <- if (r$p < .001) "***" else if (r$p < .01) "**" else if (r$p < .05) "*" else ""
    log_msg(sprintf("    %-14s  beta = %6.3f, p = %s %s",
      r$predictor, r$beta, formatC(r$p, format = "g", digits = 3), sig))
  }
}

# PART B: Parallel process LGC  rhythm + symptoms joint trajectory

log_msg("\n\n", strrep("=", 78))
log_msg("PART B: Bivariate parallel-process LGC (mesor + symptoms)")
log_msg(strrep("=", 78))
log_msg("Tests whether within-person change in mesor covaries with change in symptoms")

# Build wide mesor across waves
mesor_wide <- blups_all |>
  select(participant_id, session_id, mesor_blup) |>
  pivot_wider(id_cols = participant_id, names_from = session_id,
              values_from = mesor_blup, names_prefix = "mesor_")

pp_results <- list()

for (slug in names(scales)) {
  sc <- scales[[slug]]
  log_msg("\n", strrep("-", 40))
  log_msg("Parallel process: mesor x ", sc$label)

  cbcl_wide <- mh |>
    filter(session_id %in% lgc_waves) |>
    select(participant_id, session_id, all_of(sc$col)) |>
    filter(!is.na(.data[[sc$col]])) |>
    pivot_wider(id_cols = participant_id, names_from = session_id,
                values_from = all_of(sc$col), names_prefix = "sx_")

  pp_df <- mesor_wide |>
    inner_join(cbcl_wide, by = "participant_id") |>
    left_join(demo |> select(participant_id, sex, site_baseline), by = "participant_id") |>
    mutate(is_female = as.numeric(sex == 2))

  pp_df <- pp_df |>
    rename(
      m2 = `mesor_ses-02A`, m4 = `mesor_ses-04A`, m6 = `mesor_ses-06A`,
      s2 = `sx_ses-02A`,    s4 = `sx_ses-04A`,    s6 = `sx_ses-06A`
    )

  n_both <- sum(
    rowSums(!is.na(pp_df[c("m2","m4","m6")])) >= 2 &
    rowSums(!is.na(pp_df[c("s2","s4","s6")])) >= 2
  )
  log_msg("  N with 2+ waves both mesor and ", sc$label, ": ", n_both)

  if (n_both < 100) {
    log_msg("  SKIPPED — too few")
    next
  }

  pp_syn <- '
    # Mesor trajectory
    i_m =~ 1*m2 + 1*m4 + 1*m6
    s_m =~ 0*m2 + 1*m4 + 2*m6

    # Symptom trajectory
    i_s =~ 1*s2 + 1*s4 + 1*s6
    s_s =~ 0*s2 + 1*s4 + 2*s6

    # Correlated slopes (key test)
    s_m ~~ s_s
    i_m ~~ i_s

    # Cross-domain: mesor intercept -> symptom slope
    s_s ~ i_m

    # Control for sex
    i_m ~ is_female
    s_m ~ is_female
    i_s ~ is_female
    s_s ~ is_female
  '

  pp_fit <- tryCatch(
    growth(pp_syn, data = pp_df, missing = "fiml", estimator = "MLR",
           cluster = "site_baseline"),
    error = function(e) { log_msg("  Model failed: ", e$message); NULL }
  )
  if (is.null(pp_fit)) next

  pp_est <- parameterEstimates(pp_fit, standardized = TRUE)

  # Slope-slope correlation
  ss_cov <- pp_est |> filter(op == "~~", lhs == "s_m", rhs == "s_s")
  # Intercept-intercept correlation
  ii_cov <- pp_est |> filter(op == "~~", lhs == "i_m", rhs == "i_s")
  # Cross-lag: mesor intercept -> symptom slope
  cross <- pp_est |> filter(op == "~", lhs == "s_s", rhs == "i_m")

  if (nrow(ss_cov) > 0) {
    log_msg(sprintf("  Slope-slope cov (mesor<->%s): r = %.3f, p = %s",
      sc$label, ss_cov$std.all[1],
      formatC(ss_cov$pvalue[1], format = "g", digits = 3)))
  }
  if (nrow(ii_cov) > 0) {
    log_msg(sprintf("  Intercept-intercept cov: r = %.3f, p = %s",
      ii_cov$std.all[1],
      formatC(ii_cov$pvalue[1], format = "g", digits = 3)))
  }
  if (nrow(cross) > 0) {
    log_msg(sprintf("  Mesor intercept -> %s slope: beta = %.3f, p = %s",
      sc$label, cross$std.all[1],
      formatC(cross$pvalue[1], format = "g", digits = 3)))
  }

  fm <- fitMeasures(pp_fit, c("cfi", "rmsea"))

  pp_results[[slug]] <- tibble(
    scale = sc$label,
    n = lavInspect(pp_fit, "nobs"),
    cfi = as.numeric(fm["cfi"]),
    rmsea = as.numeric(fm["rmsea"]),
    slope_slope_r = if (nrow(ss_cov) > 0) ss_cov$std.all[1] else NA_real_,
    slope_slope_p = if (nrow(ss_cov) > 0) ss_cov$pvalue[1] else NA_real_,
    int_int_r     = if (nrow(ii_cov) > 0) ii_cov$std.all[1] else NA_real_,
    int_int_p     = if (nrow(ii_cov) > 0) ii_cov$pvalue[1] else NA_real_,
    cross_mesor_to_sx_slope_beta = if (nrow(cross) > 0) cross$std.all[1] else NA_real_,
    cross_mesor_to_sx_slope_p    = if (nrow(cross) > 0) cross$pvalue[1] else NA_real_
  )
}

pp_summary <- bind_rows(pp_results)
write_csv(pp_summary, file.path(TRAJ_DIR, "parallel_process_summary.csv"))

log_msg("\n\nParallel process summary:")
log_msg(sprintf("%-12s  %5s  %8s  %8s  %8s  %8s  %8s  %8s",
  "Scale", "N", "s-s r", "s-s p", "i-i r", "i-i p", "cross b", "cross p"))
for (i in seq_len(nrow(pp_summary))) {
  r <- pp_summary[i, ]
  log_msg(sprintf("%-12s  %5d  %8.3f  %8s  %8.3f  %8s  %8.3f  %8s",
    r$scale, r$n,
    r$slope_slope_r, formatC(r$slope_slope_p, format = "g", digits = 3),
    r$int_int_r, formatC(r$int_int_p, format = "g", digits = 3),
    r$cross_mesor_to_sx_slope_beta,
    formatC(r$cross_mesor_to_sx_slope_p, format = "g", digits = 3)))
}

writeLines(log_lines, file.path(OUT_DIR, "23_trajectory_all_dsm_and_parallel.log"))
log_msg("\nDone. Wrote log to ", file.path(OUT_DIR, "23_trajectory_all_dsm_and_parallel.log"))
