#!/usr/bin/env Rscript
# Stage 3a sensitivity: 24h + 12h harmonic cosinor mixed-effects per wave.
#
# Model:
#   y ~ rrr + sss + rrr12 + sss12 + (1 + rrr + sss + rrr12 + sss12 | id)
# with fallback hierarchy on convergence failure:
#   (1) full correlated 5-effect random structure
#   (2) diagonal (uncorrelated) 5-effect random structure (||)
#   (3) (1 + rrr + sss | id)  -- 12h randoms dropped; 12h is then a fixed
#       population-level component only (per-subject 12h BLUPs become NA)
#
# Reads:  derivatives/hourly_profiles/{wave}.parquet
# Writes: derivatives/cosinor_features/per_wave_24h_12h/{wave}/
#           participant_blups.parquet
#           population_estimates.json
#           model_diagnostics.json

suppressPackageStartupMessages({
  library(arrow); library(lme4); library(dplyr); library(jsonlite); library(tibble)
})

DERIV    <- "/Users/eu/Desktop/dairc/derivatives"
PROF_DIR <- file.path(DERIV, "hourly_profiles")
OUT_DIR  <- file.path(DERIV, "cosinor_features", "per_wave_24h_12h")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

WAVES <- c("ses-02A", "ses-04A", "ses-06A")
ctrl  <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 5e5))

fit_with_fallback <- function(forms, data, ctrl) {
  for (f in forms) {
    msg <- character(0)
    fit <- tryCatch(
      withCallingHandlers(
        lmer(f, data = data, REML = TRUE, control = ctrl),
        message = function(m) { msg <<- c(msg, conditionMessage(m)); invokeRestart("muffleMessage") },
        warning = function(w) { msg <<- c(msg, conditionMessage(w)); invokeRestart("muffleWarning") }
      ),
      error = function(e) e
    )
    if (!inherits(fit, "error") &&
        isTRUE(fit@optinfo$conv$opt == 0) &&
        length(fit@optinfo$conv$lme4$messages) == 0) {
      return(list(fit = fit, formula = paste(deparse(f), collapse = " "), warnings = msg))
    }
  }
  return(list(fit = fit, formula = paste(deparse(f), collapse = " "),
              warnings = c(msg, "all fallbacks attempted (last result returned)")))
}

# ---------------------------------------------------------------------------
# Per-wave 24h + 12h fits
# ---------------------------------------------------------------------------
all_summary <- list()

for (wave in WAVES) {
  cat(sprintf("\n=== %s : 24h + 12h harmonic ===\n", wave))
  prof <- arrow::read_parquet(file.path(PROF_DIR, sprintf("%s.parquet", wave)))
  prof <- prof %>%
    rename(id = subject_id, y = hr_median) %>%
    mutate(rrr   = cos(2*pi*clock_hour/24), sss   = sin(2*pi*clock_hour/24),
           rrr12 = cos(2*pi*clock_hour/12), sss12 = sin(2*pi*clock_hour/12))
  cat(sprintf("  n_obs = %d   n_subjects = %d\n", nrow(prof), length(unique(prof$id))))

  forms <- list(
    y ~ rrr + sss + rrr12 + sss12 + (1 + rrr + sss + rrr12 + sss12 | id),
    y ~ rrr + sss + rrr12 + sss12 + (1 + rrr + sss + rrr12 + sss12 || id),
    y ~ rrr + sss + rrr12 + sss12 + (1 + rrr + sss | id),
    y ~ rrr + sss + rrr12 + sss12 + (1 | id)
  )

  t0 <- Sys.time()
  res <- fit_with_fallback(forms, prof, ctrl)
  fit <- res$fit
  el <- as.numeric(Sys.time() - t0, units = "secs")
  cat(sprintf("  formula used: %s\n", res$formula))
  cat(sprintf("  fit time: %.1fs\n", el))

  has_h12_random <- ("rrr12" %in% names(ranef(fit)$id)) || ("sss12" %in% names(ranef(fit)$id))

  # ---- Population-level (fixed-effect) parameters ----
  fe <- fixef(fit); vc <- as.matrix(vcov(fit))
  pop_mesor <- unname(fe["(Intercept)"])
  pop_amp24 <- sqrt(fe["rrr"]^2 + fe["sss"]^2)
  pop_acr24 <- (atan2(fe["sss"], fe["rrr"]) / (2*pi) * 24) %% 24
  pop_amp12 <- sqrt(fe["rrr12"]^2 + fe["sss12"]^2)
  # 12h acrophase reported in [0,12) hours of the half-cycle
  pop_acr12 <- (atan2(fe["sss12"], fe["rrr12"]) / (2*pi) * 12) %% 12

  # Delta-method SE for amplitudes
  amp_se <- function(rrr_name, sss_name, amp) {
    g <- c(fe[rrr_name]/amp, fe[sss_name]/amp)
    sqrt(t(g) %*% vc[c(rrr_name, sss_name), c(rrr_name, sss_name)] %*% g) |> as.numeric()
  }
  se_int   <- sqrt(vc["(Intercept)","(Intercept)"])
  se_amp24 <- amp_se("rrr", "sss", pop_amp24)
  se_amp12 <- amp_se("rrr12", "sss12", pop_amp12)

  pop <- list(
    wave = wave,
    formula_used = res$formula,
    has_h12_random_effects = has_h12_random,
    n_obs = nrow(prof), n_subjects = length(unique(prof$id)),
    fixed_effects = list(
      mesor      = list(estimate = pop_mesor, se = se_int,
                        ci95 = c(pop_mesor - 1.96*se_int, pop_mesor + 1.96*se_int)),
      amplitude_24 = list(estimate = unname(pop_amp24), se = se_amp24,
                          ci95 = c(unname(pop_amp24) - 1.96*se_amp24,
                                   unname(pop_amp24) + 1.96*se_amp24)),
      acrophase_24 = list(estimate_hours = unname(pop_acr24)),
      amplitude_12 = list(estimate = unname(pop_amp12), se = se_amp12,
                          ci95 = c(unname(pop_amp12) - 1.96*se_amp12,
                                   unname(pop_amp12) + 1.96*se_amp12)),
      acrophase_12 = list(estimate_hours = unname(pop_acr12),
                          note = "Half-cycle phase, in [0, 12) hours")
    ),
    raw_fixed_coefficients = as.list(fe)
  )

  # ---- Per-participant BLUPs (5 features) ----
  re <- ranef(fit)$id
  ids <- rownames(re)
  int_dev <- if ("(Intercept)" %in% names(re)) re$"(Intercept)" else rep(0, length(ids))
  rrr_dev <- if ("rrr"   %in% names(re)) re$rrr   else rep(0, length(ids))
  sss_dev <- if ("sss"   %in% names(re)) re$sss   else rep(0, length(ids))
  r12_dev <- if ("rrr12" %in% names(re)) re$rrr12 else rep(0, length(ids))
  s12_dev <- if ("sss12" %in% names(re)) re$sss12 else rep(0, length(ids))

  rrr_total   <- unname(fe["rrr"])   + rrr_dev
  sss_total   <- unname(fe["sss"])   + sss_dev
  rrr12_total <- unname(fe["rrr12"]) + r12_dev
  sss12_total <- unname(fe["sss12"]) + s12_dev

  amp24 <- sqrt(rrr_total^2 + sss_total^2)
  acr24 <- (atan2(sss_total, rrr_total) / (2*pi) * 24) %% 24
  amp12 <- sqrt(rrr12_total^2 + sss12_total^2)
  acr12 <- (atan2(sss12_total, rrr12_total) / (2*pi) * 12) %% 12

  blups <- tibble(
    subject_id   = ids,
    wave         = wave,
    mesor_blup     = pop_mesor + int_dev,
    amplitude_24_blup = amp24,
    acrophase_24_blup = acr24,
    amplitude_12_blup = amp12,
    acrophase_12_blup = acr12
  )

  # If 12h randoms got dropped, mark per-subject 12h features as population-only.
  if (!has_h12_random) {
    # Don't NA them — they still equal the fixed-effect 12h component, useful as
    # population-anchor — but flag this in the diagnostics.
  }

  # ---- Compute per-subject R² of full 2-component model ----
  prof$pred_full <- with(prof,
                          (pop_mesor + int_dev[match(id, ids)]) +
                          rrr_total[match(id, ids)] * cos(2*pi*clock_hour/24) +
                          sss_total[match(id, ids)] * sin(2*pi*clock_hour/24) +
                          rrr12_total[match(id, ids)] * cos(2*pi*clock_hour/12) +
                          sss12_total[match(id, ids)] * sin(2*pi*clock_hour/12))
  r2 <- prof %>% group_by(id) %>%
    summarise(r_squared_2c = if (sd(y, na.rm=TRUE) > 0)
                                suppressWarnings(cor(y, pred_full, use="complete.obs"))^2
                             else NA_real_,
              n_hours_observed = sum(!is.na(y)), .groups = "drop") %>%
    rename(subject_id = id)

  blups <- blups %>% left_join(r2, by = "subject_id") %>%
    select(subject_id, wave, mesor_blup,
           amplitude_24_blup, acrophase_24_blup,
           amplitude_12_blup, acrophase_12_blup,
           r_squared_2c, n_hours_observed)

  diag <- list(
    wave = wave,
    formula_used = res$formula,
    has_h12_random_effects = has_h12_random,
    convergence_code = fit@optinfo$conv$opt,
    convergence_messages = fit@optinfo$conv$lme4$messages,
    warnings = res$warnings,
    log_likelihood = as.numeric(logLik(fit)),
    AIC = AIC(fit),
    BIC = BIC(fit),
    runtime_seconds = el
  )

  out_d <- file.path(OUT_DIR, wave); dir.create(out_d, showWarnings = FALSE, recursive = TRUE)
  arrow::write_parquet(blups, file.path(out_d, "participant_blups.parquet"))
  write_json(pop,  file.path(out_d, "population_estimates.json"), pretty = TRUE, auto_unbox = TRUE)
  write_json(diag, file.path(out_d, "model_diagnostics.json"),    pretty = TRUE, auto_unbox = TRUE)

  cat(sprintf("  pop  mesor=%.2f  amp24=%.2f  acr24=%.2f h  amp12=%.2f  acr12=%.2f h\n",
              pop_mesor, pop_amp24, pop_acr24, pop_amp12, pop_acr12))
  cat(sprintf("  R² (2c) median %.3f  IQR %.3f-%.3f\n",
              median(blups$r_squared_2c, na.rm=TRUE),
              quantile(blups$r_squared_2c, 0.25, na.rm=TRUE),
              quantile(blups$r_squared_2c, 0.75, na.rm=TRUE)))
  cat(sprintf("  AIC=%.1f   BIC=%.1f\n", AIC(fit), BIC(fit)))

  all_summary[[wave]] <- list(
    AIC_2c = AIC(fit), BIC_2c = BIC(fit), logLik_2c = as.numeric(logLik(fit)),
    has_h12_random_effects = has_h12_random,
    formula_used = res$formula
  )
}

# Compare to single-component AIC/BIC stored in v1 model_diagnostics.json
v1_summary <- list()
for (wave in WAVES) {
  d1 <- jsonlite::read_json(file.path(DERIV, "cosinor_features/per_wave", wave,
                                       "model_diagnostics.json"))
  v1_summary[[wave]] <- list(AIC_1c = d1$AIC, BIC_1c = d1$BIC, logLik_1c = d1$log_likelihood)
}

cat("\n=== ΔAIC / ΔBIC vs single-24h ===\n")
for (wave in WAVES) {
  d_aic <- all_summary[[wave]]$AIC_2c - v1_summary[[wave]]$AIC_1c
  d_bic <- all_summary[[wave]]$BIC_2c - v1_summary[[wave]]$BIC_1c
  d_ll  <- all_summary[[wave]]$logLik_2c - v1_summary[[wave]]$logLik_1c
  cat(sprintf("  %s: ΔAIC=%+.1f  ΔBIC=%+.1f  ΔlogLik=%+.1f  (negative = 2c better)\n",
              wave, d_aic, d_bic, d_ll))
}

cat("\nDone.\n")
