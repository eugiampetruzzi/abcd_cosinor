#!/usr/bin/env Rscript
# Stage 3a: per-wave + pooled cosinor mixed-effects fits.
#
# Reads:  /Users/eu/Desktop/dairc/derivatives/hourly_profiles/{ses-02A,04A,06A}.parquet
# Writes: /Users/eu/Desktop/dairc/derivatives/cosinor_features/
#           per_wave/{wave}/participant_blups.parquet
#           per_wave/{wave}/population_estimates.json
#           per_wave/{wave}/model_diagnostics.json
#           pooled/participant_blups.parquet
#           pooled/population_estimates.json
#           pooled/model_diagnostics.json
#
# lme4 is called directly (the cosinoRmixedeffects fit.cosinor.mixed wrapper
# requires fixed-effect covariates). Cosinor basis: rrr=cos(2πt/24),
# sss=sin(2πt/24). Random-effects fallback hierarchy on convergence failure:
#   (1) (1 + rrr + sss | id)   -- full
#   (2) (1 + rrr | id)         -- drop sss random
#   (3) (1 | id)               -- random mesor only

suppressPackageStartupMessages({
  library(arrow); library(lme4); library(dplyr); library(jsonlite); library(tibble)
})

DERIV    <- "/Users/eu/Desktop/dairc/derivatives"
PROF_DIR <- file.path(DERIV, "hourly_profiles")
OUT_DIR  <- file.path(DERIV, "cosinor_features")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUT_DIR, "per_wave"), showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "pooled"),   showWarnings = FALSE)

WAVES <- c("ses-02A", "ses-04A", "ses-06A")

# ---------------------------------------------------------------------------
# Utility: extract per-participant cosinor params from fixed + random effects
# ---------------------------------------------------------------------------
participant_params <- function(fit, fe_offsets = NULL) {
  # fe_offsets: optional named list of per-wave offsets to add to the base
  # fixed effects (used in pooled model). Default = base wave (ses-02A).
  fe <- fixef(fit)
  re <- ranef(fit)$id  # data frame indexed by id; cols are (Intercept), rrr, sss [or subset]

  # Default offsets = 0 (no wave correction)
  base_int <- unname(fe["(Intercept)"])
  base_rrr <- if ("rrr" %in% names(fe)) unname(fe["rrr"]) else 0
  base_sss <- if ("sss" %in% names(fe)) unname(fe["sss"]) else 0
  if (!is.null(fe_offsets)) {
    base_int <- base_int + fe_offsets[["int"]]
    base_rrr <- base_rrr + fe_offsets[["rrr"]]
    base_sss <- base_sss + fe_offsets[["sss"]]
  }

  int_dev <- if ("(Intercept)" %in% names(re)) re$"(Intercept)" else rep(0, nrow(re))
  rrr_dev <- if ("rrr" %in% names(re)) re$"rrr" else rep(0, nrow(re))
  sss_dev <- if ("sss" %in% names(re)) re$"sss" else rep(0, nrow(re))

  rrr_total <- base_rrr + rrr_dev
  sss_total <- base_sss + sss_dev
  amp <- sqrt(rrr_total^2 + sss_total^2)
  acr <- (atan2(sss_total, rrr_total) / (2*pi) * 24) %% 24
  tibble(
    subject_id = rownames(re),
    mesor_blup = base_int + int_dev,
    amplitude_blup = amp,
    acrophase_blup = acr
  )
}

# Per-participant R² of own observations given own BLUPs
participant_r2 <- function(dat, blups, fe_per_subject = NULL) {
  # fe_per_subject: a function(subject_id) -> list(int, rrr, sss) of fixed-effect totals
  # for that subject (incorporates wave fixed effects in pooled model). For per-wave model,
  # fe_per_subject = NULL and we use the global fixefs.
  d <- dat
  d$pred_int <- NA_real_; d$pred_rrr <- NA_real_; d$pred_sss <- NA_real_
  if (is.null(fe_per_subject)) {
    # Per-wave: blups already contain mesor (= int + int_dev). Build fitted by
    # directly using mesor + amplitude*cos(2π(t - acrophase)/24).
    bd <- blups
    # Match subject->row index in d
    j <- match(d$id, bd$subject_id)
    pred <- bd$mesor_blup[j] +
            bd$amplitude_blup[j] * cos(2*pi*(d$clock_hour - bd$acrophase_blup[j]) / 24)
    d$pred <- pred
  } else {
    # Pooled: subject's per-row fitted depends on row's wave too. Caller computes pred.
    stop("fe_per_subject path not used in this function; compute pred outside")
  }
  d %>%
    group_by(id) %>%
    summarise(
      r_squared = if (sd(y, na.rm=TRUE) > 0)
                    suppressWarnings(cor(y, pred, use = "complete.obs"))^2
                  else NA_real_,
      n_hours_observed = sum(!is.na(y)),
      .groups = "drop"
    )
}

fit_with_fallback <- function(form_full, form_dr1, form_dr2, data, ctrl) {
  for (form in list(form_full, form_dr1, form_dr2)) {
    msg <- character(0)
    fit <- tryCatch(
      withCallingHandlers(
        lmer(form, data = data, REML = TRUE, control = ctrl),
        message = function(m) { msg <<- c(msg, conditionMessage(m)); invokeRestart("muffleMessage") },
        warning = function(w) { msg <<- c(msg, conditionMessage(w)); invokeRestart("muffleWarning") }
      ),
      error = function(e) e
    )
    converged <- !inherits(fit, "error") &&
                 isTRUE(fit@optinfo$conv$opt == 0) &&
                 length(fit@optinfo$conv$lme4$messages) == 0
    if (converged || (!inherits(fit, "error") && !any(grepl("failed to converge", msg)))) {
      return(list(fit = fit, formula = deparse(form), warnings = msg))
    }
  }
  return(list(fit = fit, formula = deparse(form), warnings = c(msg, "all fallbacks attempted")))
}

# ---------------------------------------------------------------------------
# Per-wave fits
# ---------------------------------------------------------------------------
ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

for (wave in WAVES) {
  cat(sprintf("\n=== Per-wave: %s ===\n", wave))
  prof <- arrow::read_parquet(file.path(PROF_DIR, sprintf("%s.parquet", wave)))
  prof <- prof %>%
    rename(id = subject_id, y = hr_median) %>%
    mutate(rrr = cos(2*pi*clock_hour/24), sss = sin(2*pi*clock_hour/24))
  cat(sprintf("  n_obs = %d   n_subjects = %d\n", nrow(prof), length(unique(prof$id))))

  t0 <- Sys.time()
  res <- fit_with_fallback(
    form_full = y ~ rrr + sss + (1 + rrr + sss | id),
    form_dr1  = y ~ rrr + sss + (1 + rrr | id),
    form_dr2  = y ~ rrr + sss + (1 | id),
    data = prof, ctrl = ctrl
  )
  fit <- res$fit
  el <- as.numeric(Sys.time() - t0, units = "secs")
  cat(sprintf("  formula used: %s\n", res$formula))
  cat(sprintf("  fit time: %.1fs\n", el))

  # Population-level fixed effects + Wald CIs
  fe <- fixef(fit)
  vcov_fe <- as.matrix(vcov(fit))
  se_int <- sqrt(vcov_fe["(Intercept)", "(Intercept)"])
  se_rrr <- sqrt(vcov_fe["rrr", "rrr"])
  se_sss <- sqrt(vcov_fe["sss", "sss"])
  pop_mesor <- unname(fe["(Intercept)"])
  pop_amp   <- sqrt(fe["rrr"]^2 + fe["sss"]^2)
  pop_acr   <- (atan2(fe["sss"], fe["rrr"]) / (2*pi) * 24) %% 24
  # Delta-method SE for amplitude (acrophase delta-method skipped; report bootstrap-able point)
  cov_rs <- vcov_fe["rrr","sss"]
  amp_grad <- c(fe["rrr"]/pop_amp, fe["sss"]/pop_amp)
  amp_se   <- sqrt(t(amp_grad) %*% vcov_fe[c("rrr","sss"), c("rrr","sss")] %*% amp_grad) |> as.numeric()

  pop <- list(
    wave = wave,
    n_obs = nrow(prof),
    n_subjects = length(unique(prof$id)),
    fixed_effects = list(
      mesor = list(estimate = pop_mesor, se = se_int,
                   ci95 = c(pop_mesor - 1.96*se_int, pop_mesor + 1.96*se_int)),
      amplitude = list(estimate = unname(pop_amp), se = amp_se,
                       ci95 = c(unname(pop_amp) - 1.96*amp_se, unname(pop_amp) + 1.96*amp_se)),
      acrophase = list(estimate_hours = unname(pop_acr),
                        note = "Wald CI on acrophase requires circular delta method; bootstrap recommended for inference.")
    ),
    raw_fixed_coefficients = list(
      intercept = pop_mesor, rrr = unname(fe["rrr"]), sss = unname(fe["sss"])
    )
  )

  # Per-participant BLUPs
  blups <- participant_params(fit)
  # R²
  r2 <- participant_r2(prof, blups)
  blups <- blups %>% left_join(r2, by = c("subject_id" = "id")) %>%
           mutate(wave = wave) %>%
           select(subject_id, wave, mesor_blup, amplitude_blup, acrophase_blup, r_squared, n_hours_observed)

  # Diagnostics
  diag <- list(
    wave = wave,
    formula_used = res$formula,
    convergence_code = fit@optinfo$conv$opt,
    convergence_messages = fit@optinfo$conv$lme4$messages,
    warnings = res$warnings,
    log_likelihood = as.numeric(logLik(fit)),
    AIC = AIC(fit),
    BIC = BIC(fit),
    runtime_seconds = el
  )

  out_d <- file.path(OUT_DIR, "per_wave", wave)
  dir.create(out_d, showWarnings = FALSE, recursive = TRUE)
  arrow::write_parquet(blups, file.path(out_d, "participant_blups.parquet"))
  write_json(pop,  file.path(out_d, "population_estimates.json"), pretty = TRUE, auto_unbox = TRUE)
  write_json(diag, file.path(out_d, "model_diagnostics.json"),    pretty = TRUE, auto_unbox = TRUE)

  cat(sprintf("  pop  mesor=%.2f  amp=%.2f  acrophase=%.2f h\n",
              pop_mesor, pop_amp, pop_acr))
  cat(sprintf("  BLUPs: median mesor=%.2f, median amp=%.2f, median acrophase=%.2f h\n",
              median(blups$mesor_blup), median(blups$amplitude_blup),
              median(blups$acrophase_blup)))
  cat(sprintf("  R² median %.3f  IQR %.3f-%.3f  n<0.10: %d / %d\n",
              median(blups$r_squared, na.rm=TRUE),
              quantile(blups$r_squared, 0.25, na.rm=TRUE),
              quantile(blups$r_squared, 0.75, na.rm=TRUE),
              sum(blups$r_squared < 0.10, na.rm=TRUE),
              sum(!is.na(blups$r_squared))))
}

# ---------------------------------------------------------------------------
# Pooled model: per-wave fixed-effect contrasts; per-subject random rhythm
# ---------------------------------------------------------------------------
cat("\n=== Pooled (sensitivity) ===\n")
prof_all <- bind_rows(lapply(WAVES, function(w) {
  arrow::read_parquet(file.path(PROF_DIR, sprintf("%s.parquet", w))) %>%
    rename(id = subject_id, y = hr_median) %>%
    mutate(wave = w,
           rrr = cos(2*pi*clock_hour/24),
           sss = sin(2*pi*clock_hour/24))
}))
prof_all$wave <- factor(prof_all$wave, levels = WAVES)
cat(sprintf("  n_obs = %d   n_subjects = %d\n", nrow(prof_all), length(unique(prof_all$id))))

t0 <- Sys.time()
res_p <- fit_with_fallback(
  form_full = y ~ wave + rrr + sss + wave:rrr + wave:sss + (1 + rrr + sss | id),
  form_dr1  = y ~ wave + rrr + sss + wave:rrr + wave:sss + (1 + rrr | id),
  form_dr2  = y ~ wave + rrr + sss + wave:rrr + wave:sss + (1 | id),
  data = prof_all, ctrl = ctrl
)
fit_p <- res_p$fit
el_p <- as.numeric(Sys.time() - t0, units = "secs")
cat(sprintf("  formula used: %s\n", res_p$formula))
cat(sprintf("  fit time: %.1fs\n", el_p))

fe_p <- fixef(fit_p)
print(fe_p)

# Build per-wave fixed-effect totals from the pooled model
wave_offsets <- list(
  "ses-02A" = list(int = 0, rrr = 0, sss = 0),
  "ses-04A" = list(int = unname(fe_p["waveses-04A"]),
                    rrr = unname(fe_p["waveses-04A:rrr"]),
                    sss = unname(fe_p["waveses-04A:sss"])),
  "ses-06A" = list(int = unname(fe_p["waveses-06A"]),
                    rrr = unname(fe_p["waveses-06A:rrr"]),
                    sss = unname(fe_p["waveses-06A:sss"]))
)
# Per-wave population params from pooled
pooled_pop <- lapply(WAVES, function(w) {
  off <- wave_offsets[[w]]
  m <- unname(fe_p["(Intercept)"]) + off$int
  rt <- unname(fe_p["rrr"]) + off$rrr
  st <- unname(fe_p["sss"]) + off$sss
  amp <- sqrt(rt^2 + st^2); acr <- (atan2(st, rt)/(2*pi)*24) %% 24
  list(wave = w, mesor = m, amplitude = amp, acrophase_hours = acr)
})
names(pooled_pop) <- WAVES

# Per-subject BLUPs (constant across waves: deviation from base wave intercept).
# To produce one row per (subject, wave), expand and apply each wave's offsets.
re_p <- ranef(fit_p)$id
int_dev <- if ("(Intercept)" %in% names(re_p)) re_p$"(Intercept)" else rep(0, nrow(re_p))
rrr_dev <- if ("rrr" %in% names(re_p)) re_p$"rrr" else rep(0, nrow(re_p))
sss_dev <- if ("sss" %in% names(re_p)) re_p$"sss" else rep(0, nrow(re_p))
sub_ids <- rownames(re_p)

blups_p <- bind_rows(lapply(WAVES, function(w) {
  off <- wave_offsets[[w]]
  m <- unname(fe_p["(Intercept)"]) + off$int + int_dev
  rt <- unname(fe_p["rrr"]) + off$rrr + rrr_dev
  st <- unname(fe_p["sss"]) + off$sss + sss_dev
  amp <- sqrt(rt^2 + st^2); acr <- (atan2(st, rt)/(2*pi)*24) %% 24
  tibble(subject_id = sub_ids, wave = w,
         mesor_blup = m, amplitude_blup = amp, acrophase_blup = acr)
}))
# Filter to subjects who actually contributed at that wave
present <- prof_all %>% distinct(id, wave) %>% rename(subject_id = id)
blups_p <- blups_p %>% inner_join(present, by = c("subject_id","wave"))

# Per-(subject,wave) R²
prof_all <- prof_all %>%
  mutate(pred = NA_real_)
for (w in WAVES) {
  off <- wave_offsets[[w]]
  ix <- prof_all$wave == w
  bd <- blups_p %>% filter(wave == w)
  j <- match(prof_all$id[ix], bd$subject_id)
  prof_all$pred[ix] <- bd$mesor_blup[j] +
                       bd$amplitude_blup[j] * cos(2*pi*(prof_all$clock_hour[ix] - bd$acrophase_blup[j])/24)
}
r2_p <- prof_all %>% group_by(id, wave) %>%
  summarise(r_squared = if (sd(y, na.rm=TRUE) > 0)
                          suppressWarnings(cor(y, pred, use="complete.obs"))^2 else NA_real_,
            n_hours_observed = sum(!is.na(y)),
            .groups = "drop") %>%
  rename(subject_id = id)
blups_p <- blups_p %>% left_join(r2_p, by = c("subject_id","wave")) %>%
  select(subject_id, wave, mesor_blup, amplitude_blup, acrophase_blup, r_squared, n_hours_observed)

pop_p <- list(
  formula_used = res_p$formula,
  n_obs = nrow(prof_all),
  n_subjects = length(unique(prof_all$id)),
  per_wave_population = pooled_pop,
  raw_fixed_coefficients = as.list(fe_p)
)
diag_p <- list(
  formula_used = res_p$formula,
  convergence_code = fit_p@optinfo$conv$opt,
  convergence_messages = fit_p@optinfo$conv$lme4$messages,
  warnings = res_p$warnings,
  log_likelihood = as.numeric(logLik(fit_p)),
  AIC = AIC(fit_p),
  BIC = BIC(fit_p),
  runtime_seconds = el_p
)
out_p <- file.path(OUT_DIR, "pooled")
arrow::write_parquet(blups_p, file.path(out_p, "participant_blups.parquet"))
write_json(pop_p,  file.path(out_p, "population_estimates.json"), pretty=TRUE, auto_unbox=TRUE)
write_json(diag_p, file.path(out_p, "model_diagnostics.json"),    pretty=TRUE, auto_unbox=TRUE)

cat(sprintf("  pooled rows in BLUP file: %d (one per subject*wave actually present)\n", nrow(blups_p)))
cat(sprintf("  pooled R² median %.3f, IQR %.3f-%.3f\n",
            median(blups_p$r_squared, na.rm=TRUE),
            quantile(blups_p$r_squared, 0.25, na.rm=TRUE),
            quantile(blups_p$r_squared, 0.75, na.rm=TRUE)))
cat("\nDone.\n")
