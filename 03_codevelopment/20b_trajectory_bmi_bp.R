# 20b - Incremental prediction of BMI and blood pressure trajectories
# Parallel to script 20 (depression), fits latent growth curves for:
# (A) BMI across Waves 2/4/6 (baseline W0 BMI as external covariate)
# (B) Systolic BP across Waves 2/4/6 (W0 BMI as proxy baseline covariate;
# no W0 BP measurement exists in ABCD)
# Same hierarchical block entry:
# M0  Unconditional LGC
# M1  + demographics (sex, race/ethnicity, income, parental education)
# M2  + baseline (W0 BMI z-score)
# M3  + behavioral wearables (W2 sleep, MVPA)
# M4  + cardiac rhythm (W2 mesor, amplitude, acrophase)
# Site-clustered SEs, FIML for missing waves.
# Also includes Layer 2 (single-timepoint linear regression) for each outcome.

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

z_scale <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)

# Load data

log_msg("Loading data sources...")

blups <- read_parquet(
  file.path(DERIV, "cosinor_features/per_wave/ses-02A/participant_blups.parquet")
) |>
  rename(participant_id = subject_id) |>
  select(participant_id, mesor_blup, amplitude_blup, acrophase_blup)
log_msg("  Cosinor BLUPs: n = ", nrow(blups))

ph <- read_parquet(file.path(PAPER, "outcomes/master_outcomes_physical_health.parquet"))
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

# BMI wide

waves_bmi <- c("ses-00A", "ses-02A", "ses-04A", "ses-06A")
bmi_wide <- ph |>
  filter(session_id %in% waves_bmi) |>
  select(participant_id, session_id, bmi) |>
  filter(!is.na(bmi)) |>
  pivot_wider(id_cols = participant_id, names_from = session_id,
              values_from = bmi, names_prefix = "bmi_")
log_msg("  BMI (any wave): n = ", nrow(bmi_wide))

# BP wide (W2/W4/W6 only)

waves_bp <- c("ses-02A", "ses-04A", "ses-06A")
bp_wide <- ph |>
  filter(session_id %in% waves_bp) |>
  select(participant_id, session_id, bp_sys_mean) |>
  filter(!is.na(bp_sys_mean)) |>
  pivot_wider(id_cols = participant_id, names_from = session_id,
              values_from = bp_sys_mean, names_prefix = "sbp_")
log_msg("  Systolic BP (any wave): n = ", nrow(bp_wide))

# Merge

df <- blups |>
  left_join(bmi_wide, by = "participant_id") |>
  left_join(bp_wide, by = "participant_id") |>
  left_join(demo, by = "participant_id") |>
  left_join(beh, by = "participant_id") |>
  left_join(fam, by = "participant_id")
log_msg("Merged: n = ", nrow(df))

# Prepare variables

df <- df |>
  rename(
    bmi_w0 = `bmi_ses-00A`, bmi_w2 = `bmi_ses-02A`,
    bmi_w4 = `bmi_ses-04A`, bmi_w6 = `bmi_ses-06A`,
    sbp_w2 = `sbp_ses-02A`, sbp_w4 = `sbp_ses-04A`, sbp_w6 = `sbp_ses-06A`
  )

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
df$bmi_w0_z    <- z_scale(df$bmi_w0)

demo_vars <- c("is_female",
               "race_hispanic", "race_black", "race_asian", "race_other",
               "income_lt50k", "income_50_100k",
               "edu_lt_hs", "edu_hs", "edu_some_col", "edu_bachelors")
baseline_vars <- c("bmi_w0_z")
behavioral_vars <- c("sleep_z", "mvpa_z")
rhythm_vars <- c("mesor_z", "amplitude_z", "acrophase_z")

# LGC helpers

build_lgc <- function(y2, y4, y6, covariates = NULL) {
  syn <- paste0('
    intercept =~ 1*', y2, ' + 1*', y4, ' + 1*', y6, '
    slope     =~ 0*', y2, ' + 1*', y4, ' + 2*', y6, '
  ')
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
  fm <- fitMeasures(fit, c("chisq", "df", "pvalue",
                           "cfi", "tli", "rmsea", "rmsea.ci.lower",
                           "rmsea.ci.upper", "srmr",
                           "aic", "bic", "logl", "npar"))
  r2 <- tryCatch(lavInspect(fit, "r-square"), error = function(e) c())
  tibble(
    model = model_name,
    n = lavInspect(fit, "nobs"),
    n_par = as.numeric(fm["npar"]),
    chisq = as.numeric(fm["chisq"]),
    df = as.numeric(fm["df"]),
    cfi = as.numeric(fm["cfi"]),
    tli = as.numeric(fm["tli"]),
    rmsea = as.numeric(fm["rmsea"]),
    rmsea_lo = as.numeric(fm["rmsea.ci.lower"]),
    rmsea_hi = as.numeric(fm["rmsea.ci.upper"]),
    srmr = as.numeric(fm["srmr"]),
    aic = as.numeric(fm["aic"]),
    bic = as.numeric(fm["bic"]),
    logl = as.numeric(fm["logl"]),
    r2_intercept = if ("intercept" %in% names(r2)) r2["intercept"] else NA_real_,
    r2_slope = if ("slope" %in% names(r2)) r2["slope"] else NA_real_
  )
}

run_lgc_hierarchy <- function(data, y2, y4, y6, outcome_label,
                              demo_v, base_v, beh_v, rhy_v) {
  log_msg("\n", strrep("=", 78))
  log_msg("LGC: ", outcome_label, " (", y2, ", ", y4, ", ", y6, ")")
  log_msg(strrep("=", 78))

  n_any <- sum(rowSums(!is.na(data[c(y2, y4, y6)])) >= 1)
  n_2plus <- sum(rowSums(!is.na(data[c(y2, y4, y6)])) >= 2)
  n_3 <- sum(rowSums(!is.na(data[c(y2, y4, y6)])) == 3)
  log_msg("  Available: any wave = ", n_any,
          ", 2+ waves = ", n_2plus, ", 3 waves = ", n_3)

  fits <- list()
  summaries <- list()
  blocks <- list(
    M0 = NULL,
    M1 = demo_v,
    M2 = c(demo_v, base_v),
    M3 = c(demo_v, base_v, beh_v),
    M4 = c(demo_v, base_v, beh_v, rhy_v)
  )
  labels <- c("M0_unconditional", "M1_demographics",
              "M2_+baseline", "M3_+behavior", "M4_+rhythm")

  for (i in seq_along(blocks)) {
    nm <- names(blocks)[i]
    lbl <- labels[i]
    log_msg("\n  ", lbl, "...")
    syn <- build_lgc(y2, y4, y6, covariates = blocks[[i]])
    fits[[nm]] <- tryCatch(
      growth(syn, data = data, missing = "fiml", estimator = "MLR",
             cluster = "site_baseline"),
      error = function(e) { log_msg("    FAILED: ", e$message); NULL }
    )
    if (is.null(fits[[nm]])) {
      summaries[[nm]] <- tibble(model = lbl, n = NA_integer_,
        n_par = NA, chisq = NA, df = NA, cfi = NA, tli = NA,
        rmsea = NA, rmsea_lo = NA, rmsea_hi = NA, srmr = NA,
        aic = NA, bic = NA, logl = NA,
        r2_intercept = NA, r2_slope = NA)
      next
    }
    summaries[[nm]] <- extract_fit(fits[[nm]], lbl)
    s <- summaries[[nm]]
    log_msg(sprintf("    n=%d  CFI=%.3f  RMSEA=%.3f  R2(slope)=%s",
      s$n, s$cfi, s$rmsea,
      if (is.na(s$r2_slope)) "NA" else sprintf("%.4f", s$r2_slope)))
  }

  fit_table <- bind_rows(summaries)
  fit_table$delta_r2_slope <- c(NA, diff(fit_table$r2_slope))

  # LRT
  lrt_results <- list()
  pairs <- list(c("M0","M1"), c("M1","M2"), c("M2","M3"), c("M3","M4"))
  pair_labels <- c("M0 vs M1", "M1 vs M2", "M2 vs M3", "M3 vs M4")
  for (j in seq_along(pairs)) {
    p <- pairs[[j]]
    if (is.null(fits[[p[1]]]) || is.null(fits[[p[2]]])) {
      lrt_results[[j]] <- tibble(comparison = pair_labels[j],
        reduced = p[1], full = p[2],
        chisq_diff = NA, df_diff = NA, p = NA)
      next
    }
    tryCatch({
      lrt <- lavTestLRT(fits[[p[1]]], fits[[p[2]]])
      lrt_results[[j]] <- tibble(
        comparison = pair_labels[j], reduced = p[1], full = p[2],
        chisq_diff = lrt$`Chisq diff`[2],
        df_diff = lrt$`Df diff`[2],
        p = lrt$`Pr(>Chisq)`[2]
      )
    }, error = function(e) {
      log_msg("    LRT ", pair_labels[j], " failed: ", e$message)
      lrt_results[[j]] <<- tibble(comparison = pair_labels[j],
        reduced = p[1], full = p[2],
        chisq_diff = NA, df_diff = NA, p = NA)
    })
  }
  lrt_table <- bind_rows(lrt_results)

  log_msg("\n  Model comparison:")
  for (i in seq_len(nrow(fit_table))) {
    r <- fit_table[i, ]
    log_msg(sprintf("    %-30s  n=%-5s  CFI=%-6s  R2(slope)=%-8s  dR2=%-8s",
      r$model,
      if (is.na(r$n)) "---" else as.character(r$n),
      if (is.na(r$cfi)) "---" else sprintf("%.3f", r$cfi),
      if (is.na(r$r2_slope)) "---" else sprintf("%.4f", r$r2_slope),
      if (is.na(r$delta_r2_slope)) "---" else sprintf("%.4f", r$delta_r2_slope)))
  }
  log_msg("\n  LRT:")
  for (i in seq_len(nrow(lrt_table))) {
    r <- lrt_table[i, ]
    log_msg(sprintf("    %-12s  chi2(%s) = %s, p = %s",
      r$comparison,
      if (is.na(r$df_diff)) "?" else as.character(r$df_diff),
      if (is.na(r$chisq_diff)) "---" else sprintf("%.2f", r$chisq_diff),
      if (is.na(r$p)) "---" else formatC(r$p, format = "g", digits = 3)))
  }

  # Rhythm coefficients from M4
  if (!is.null(fits$M4)) {
    m4_est <- parameterEstimates(fits$M4, standardized = TRUE)
    rhy_slope <- m4_est |> filter(op == "~", lhs == "slope", rhs %in% rhy_v)
    rhy_int   <- m4_est |> filter(op == "~", lhs == "intercept", rhs %in% rhy_v)
    log_msg("\n  M4 rhythm -> slope:")
    for (i in seq_len(nrow(rhy_slope))) {
      r <- rhy_slope[i, ]
      log_msg(sprintf("    slope ~ %-14s  b = %7.4f, beta = %.3f, p = %s",
        r$rhs, r$est, r$std.all,
        formatC(r$pvalue, format = "g", digits = 3)))
    }
    log_msg("  M4 rhythm -> intercept:")
    for (i in seq_len(nrow(rhy_int))) {
      r <- rhy_int[i, ]
      log_msg(sprintf("    intercept ~ %-14s  b = %7.4f, beta = %.3f, p = %s",
        r$rhs, r$est, r$std.all,
        formatC(r$pvalue, format = "g", digits = 3)))
    }
  }

  list(fit_table = fit_table, lrt_table = lrt_table, fits = fits,
       outcome = outcome_label)
}

# Layer 2 helper

run_lm_hierarchy <- function(data, outcome_col, baseline_col,
                             demo_v, beh_v, rhy_v, label) {
  log_msg("\n  Layer 2: ", label)
  data$outcome <- data[[outcome_col]]
  data$baseline <- data[[baseline_col]]

  demo_str <- paste(demo_v, collapse = " + ")
  results <- list()

  f1 <- as.formula(paste("outcome ~", demo_str))
  m1 <- lm(f1, data = data)
  results$M1 <- tibble(model = "M1_demographics",
    r2 = summary(m1)$r.squared, adj_r2 = summary(m1)$adj.r.squared,
    n = nobs(m1))

  f2 <- as.formula(paste("outcome ~", demo_str, "+ baseline"))
  m2 <- lm(f2, data = data)
  results$M2 <- tibble(model = "M2_+baseline",
    r2 = summary(m2)$r.squared, adj_r2 = summary(m2)$adj.r.squared,
    n = nobs(m2))

  f3 <- as.formula(paste("outcome ~", demo_str, "+ baseline +",
                         paste(beh_v, collapse = " + ")))
  m3 <- lm(f3, data = data)
  results$M3 <- tibble(model = "M3_+behavior",
    r2 = summary(m3)$r.squared, adj_r2 = summary(m3)$adj.r.squared,
    n = nobs(m3))

  f4 <- as.formula(paste("outcome ~", demo_str, "+ baseline +",
                         paste(c(beh_v, rhy_v), collapse = " + ")))
  m4 <- lm(f4, data = data)
  results$M4 <- tibble(model = "M4_+rhythm",
    r2 = summary(m4)$r.squared, adj_r2 = summary(m4)$adj.r.squared,
    n = nobs(m4))

  f12 <- anova(m1, m2); f23 <- anova(m2, m3); f34 <- anova(m3, m4)
  tbl <- bind_rows(results)
  tbl$delta_r2 <- c(NA, diff(tbl$r2))
  tbl$f_change_p <- c(NA, f12$`Pr(>F)`[2], f23$`Pr(>F)`[2], f34$`Pr(>F)`[2])

  for (i in seq_len(nrow(tbl))) {
    r <- tbl[i, ]
    log_msg(sprintf("    %-22s  n=%d  R2=%.4f  dR2=%-8s  F-change p=%s",
      r$model, r$n, r$r2,
      if (is.na(r$delta_r2)) "---" else sprintf("%.4f", r$delta_r2),
      if (is.na(r$f_change_p)) "---" else formatC(r$f_change_p, format="g", digits=3)))
  }

  m4_coefs <- summary(m4)$coefficients
  for (rv in rhy_v) {
    if (rv %in% rownames(m4_coefs)) {
      log_msg(sprintf("    M4 %s: b = %.3f (SE = %.3f), t = %.2f, p = %s",
        rv, m4_coefs[rv,1], m4_coefs[rv,2], m4_coefs[rv,3],
        formatC(m4_coefs[rv,4], format="g", digits=3)))
    }
  }
  tbl
}

# (A) BMI TRAJECTORY

bmi_result <- run_lgc_hierarchy(
  data = df, y2 = "bmi_w2", y4 = "bmi_w4", y6 = "bmi_w6",
  outcome_label = "BMI",
  demo_v = demo_vars, base_v = baseline_vars,
  beh_v = behavioral_vars, rhy_v = rhythm_vars
)

# Unconditional growth summary for BMI
if (!is.null(bmi_result$fits$M0)) {
  m0_est <- parameterEstimates(bmi_result$fits$M0, standardized = TRUE)
  m0_means <- m0_est |> filter(op == "~1", lhs %in% c("intercept", "slope"))
  m0_vars  <- m0_est |> filter(op == "~~", lhs == rhs, lhs %in% c("intercept", "slope"))
  log_msg("\n  BMI unconditional growth:")
  for (i in seq_len(nrow(m0_means))) {
    log_msg("    Mean ", m0_means$lhs[i], " = ", round(m0_means$est[i], 2),
            " (SE = ", round(m0_means$se[i], 3), ")")
  }
  for (i in seq_len(nrow(m0_vars))) {
    log_msg("    Var  ", m0_vars$lhs[i], " = ", round(m0_vars$est[i], 2))
  }
}

# Layer 2: BMI W4
log_msg("\n", strrep("=", 78))
log_msg("Layer 2: BMI single-timepoint")
log_msg(strrep("=", 78))

bmi_l2_df <- df |> filter(!is.na(bmi_w4), !is.na(bmi_w0_z),
                           !is.na(sleep_z), !is.na(mvpa_z))
log_msg("  W4 complete: n = ", nrow(bmi_l2_df))
bmi_l2_w4 <- run_lm_hierarchy(bmi_l2_df, "bmi_w4", "bmi_w0_z",
                               demo_vars, behavioral_vars, rhythm_vars,
                               "BMI W2 rhythm -> W4 BMI")

bmi_l2b_df <- df |> filter(!is.na(bmi_w6), !is.na(bmi_w0_z),
                            !is.na(sleep_z), !is.na(mvpa_z))
log_msg("  W6 complete: n = ", nrow(bmi_l2b_df))
bmi_l2_w6 <- run_lm_hierarchy(bmi_l2b_df, "bmi_w6", "bmi_w0_z",
                               demo_vars, behavioral_vars, rhythm_vars,
                               "BMI W2 rhythm -> W6 BMI")

# (B) SYSTOLIC BP TRAJECTORY

bp_result <- run_lgc_hierarchy(
  data = df, y2 = "sbp_w2", y4 = "sbp_w4", y6 = "sbp_w6",
  outcome_label = "Systolic BP",
  demo_v = demo_vars, base_v = baseline_vars,
  beh_v = behavioral_vars, rhy_v = rhythm_vars
)

if (!is.null(bp_result$fits$M0)) {
  m0_est <- parameterEstimates(bp_result$fits$M0, standardized = TRUE)
  m0_means <- m0_est |> filter(op == "~1", lhs %in% c("intercept", "slope"))
  m0_vars  <- m0_est |> filter(op == "~~", lhs == rhs, lhs %in% c("intercept", "slope"))
  log_msg("\n  Systolic BP unconditional growth:")
  for (i in seq_len(nrow(m0_means))) {
    log_msg("    Mean ", m0_means$lhs[i], " = ", round(m0_means$est[i], 2),
            " (SE = ", round(m0_means$se[i], 3), ")")
  }
  for (i in seq_len(nrow(m0_vars))) {
    log_msg("    Var  ", m0_vars$lhs[i], " = ", round(m0_vars$est[i], 2))
  }
}

# Layer 2: BP W4
log_msg("\n", strrep("=", 78))
log_msg("Layer 2: Systolic BP single-timepoint")
log_msg(strrep("=", 78))

bp_l2_df <- df |> filter(!is.na(sbp_w4), !is.na(bmi_w0_z),
                          !is.na(sleep_z), !is.na(mvpa_z))
log_msg("  W4 complete: n = ", nrow(bp_l2_df))
bp_l2_w4 <- run_lm_hierarchy(bp_l2_df, "sbp_w4", "bmi_w0_z",
                              demo_vars, behavioral_vars, rhythm_vars,
                              "SBP W2 rhythm -> W4 SBP")

bp_l2b_df <- df |> filter(!is.na(sbp_w6), !is.na(bmi_w0_z),
                           !is.na(sleep_z), !is.na(mvpa_z))
log_msg("  W6 complete: n = ", nrow(bp_l2b_df))
bp_l2_w6 <- run_lm_hierarchy(bp_l2b_df, "sbp_w6", "bmi_w0_z",
                              demo_vars, behavioral_vars, rhythm_vars,
                              "SBP W2 rhythm -> W6 SBP")

# Save outputs

save_outcome <- function(result, l2_w4, l2_w6, prefix) {
  result$fit_table$outcome <- prefix
  write_csv(result$fit_table, file.path(TRAJ_DIR,
    paste0(prefix, "_lgc_fit_comparison.csv")))
  result$lrt_table$outcome <- prefix
  write_csv(result$lrt_table, file.path(TRAJ_DIR,
    paste0(prefix, "_lgc_lrt.csv")))
  if (!is.null(result$fits$M4)) {
    m4_est <- parameterEstimates(result$fits$M4, standardized = TRUE)
    write_csv(m4_est, file.path(TRAJ_DIR,
      paste0(prefix, "_lgc_m4_estimates.csv")))
  }
  l2_w4$outcome <- prefix
  write_csv(l2_w4, file.path(TRAJ_DIR,
    paste0(prefix, "_layer2_w4.csv")))
  l2_w6$outcome <- prefix
  write_csv(l2_w6, file.path(TRAJ_DIR,
    paste0(prefix, "_layer2_w6.csv")))
  log_msg("Wrote ", prefix, " output files to ", TRAJ_DIR)
}

save_outcome(bmi_result, bmi_l2_w4, bmi_l2_w6, "bmi")
save_outcome(bp_result, bp_l2_w4, bp_l2_w6, "sbp")

writeLines(log_lines, file.path(OUT_DIR, "20b_trajectory_bmi_bp.log"))
log_msg("\nDone.")
