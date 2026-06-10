#!/usr/bin/env Rscript
# Paper 2  Analysis 1: GAMLSS normative centile curves
# Fits age x sex centile models (P3-P97) for MESOR, amplitude, acrophase.
# Reads pooled BLUPs from stdin (TSV) written by the Python wrapper.
# Writes outputs to derivatives/paper2_normative_centiles/.

suppressPackageStartupMessages({
  library(gamlss)
  library(arrow)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
deriv_dir <- args[1]
out_dir   <- file.path(deriv_dir, "paper2_normative_centiles")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

waves <- c("ses-02A", "ses-04A", "ses-06A")
blup_frames <- lapply(waves, function(w) {
  p <- file.path(deriv_dir, "cosinor_features", "per_wave", w, "participant_blups.parquet")
  d <- as.data.frame(read_parquet(p))
  d$session_id <- w
  d
})
blups <- do.call(rbind, blup_frames)
names(blups)[names(blups) == "subject_id"] <- "participant_id"

onedrive <- file.path(
  "/Users/eu/Library/CloudStorage/OneDrive-Stanford",
  "Research Projects/1 - Data/ABCD/ABCD Actigraphy Resource Paper"
)
mh <- as.data.frame(read_parquet(file.path(onedrive, "outcomes",
                                           "master_outcomes_mental_health.parquet")))
age <- mh[mh$session_id %in% waves, c("participant_id", "session_id", "cbcl_age")]
names(age)[names(age) == "cbcl_age"] <- "age_yrs"

demo_dir <- file.path(
  "/Users/eu/Library/CloudStorage/OneDrive-Stanford",
  "Research Projects/1 - Data/ABCD/Release 6.1/Demographics/phenotype"
)
stc <- read.csv(file.path(demo_dir, "ab_g_stc.tsv"), sep = "\t",
                na.strings = c("n/a", ""))
sex_df <- data.frame(
  participant_id = stc$participant_id,
  sex = ifelse(stc$ab_g_stc__cohort_sex == 2, "female",
         ifelse(stc$ab_g_stc__cohort_sex == 1, "male", NA))
)

df <- merge(blups, age, by = c("participant_id", "session_id"))
df <- merge(df, sex_df, by = "participant_id")
df <- df[complete.cases(df[, c("mesor_blup", "amplitude_blup", "acrophase_blup",
                                "age_yrs", "sex")]), ]
cat(sprintf("N = %d observations, %d unique participants\n",
            nrow(df), length(unique(df$participant_id))))

params <- list(
  list(col = "mesor_blup",     label = "MESOR"),
  list(col = "amplitude_blup", label = "Amplitude"),
  list(col = "acrophase_blup", label = "Acrophase")
)
sexes  <- c("male", "female")
cents  <- c(3, 10, 25, 50, 75, 90, 97)
families <- list(BCT = BCT, BCCG = BCCG, NO = NO)

age_seq <- seq(10, 18, by = 0.1)

dist_selection_rows <- list()
centile_frames <- list()

set.seed(20260430)

for (par in params) {
  for (sx in sexes) {
    sub <- df[df$sex == sx, ]
    y <- sub[[par$col]]
    age_v <- sub$age_yrs

    if (par$col == "acrophase_blup") {
      y <- ifelse(y < 6, y + 24, y)
    }

    best_gaic <- Inf
    best_fam  <- NULL
    best_mod  <- NULL

    for (fam_name in names(families)) {
      fam_fn <- families[[fam_name]]
      mod <- tryCatch({
        if (fam_name == "NO") {
          gamlss(y ~ pb(age_v),
                 sigma.formula = ~ pb(age_v),
                 family = fam_fn,
                 data = data.frame(y = y, age_v = age_v),
                 control = gamlss.control(n.cyc = 100, trace = FALSE))
        } else {
          gamlss(y ~ pb(age_v),
                 sigma.formula = ~ pb(age_v),
                 nu.formula = ~ pb(age_v),
                 tau.formula = ~ pb(age_v),
                 family = fam_fn,
                 data = data.frame(y = y, age_v = age_v),
                 control = gamlss.control(n.cyc = 100, trace = FALSE))
        }
      }, error = function(e) {
        cat(sprintf("  [WARN] %s/%s/%s failed: %s\n", par$label, sx, fam_name, e$message))
        NULL
      })

      if (!is.null(mod)) {
        g <- GAIC(mod, k = log(nrow(sub)))
        dist_selection_rows[[length(dist_selection_rows) + 1]] <- data.frame(
          param = par$label, sex = sx, family = fam_name,
          gaic = round(g, 2), df_used = mod$df.fit,
          stringsAsFactors = FALSE
        )
        if (g < best_gaic) {
          best_gaic <- g
          best_fam  <- fam_name
          best_mod  <- mod
        }
      }
    }

    cat(sprintf("  %s / %s: best family = %s (GAIC = %.1f)\n",
                par$label, sx, best_fam, best_gaic))

    newdata <- data.frame(age_v = age_seq)
    cp <- centiles.pred(best_mod, xname = "age_v", xvalues = age_seq,
                        cent = cents, plot = FALSE)

    cp_names <- names(cp)
    val_cols <- cp_names[cp_names != "x"]
    for (i in seq_along(cents)) {
      col_name <- val_cols[i]
      vals <- cp[[col_name]]
      if (par$col == "acrophase_blup") vals <- vals %% 24
      centile_frames[[length(centile_frames) + 1]] <- data.frame(
        param = par$label, sex = sx, centile = cents[i],
        age_yrs = cp$x,
        value = vals,
        family = best_fam,
        stringsAsFactors = FALSE
      )
    }
  }
}

dist_sel <- do.call(rbind, dist_selection_rows)
write.csv(dist_sel, file.path(out_dir, "distribution_selection.tsv"),
          row.names = FALSE)

centiles_all <- do.call(rbind, centile_frames)
for (par in params) {
  for (sx in sexes) {
    sub <- centiles_all[centiles_all$param == par$label & centiles_all$sex == sx, ]
    fn <- sprintf("centiles_%s_%s.tsv", tolower(par$label), sx)
    write.csv(sub, file.path(out_dir, fn), row.names = FALSE)
  }
}
write.csv(centiles_all, file.path(out_dir, "centiles_all.tsv"), row.names = FALSE)

# Lookup table: 0.5-yr bins
age_lookup <- seq(10, 18, by = 0.5)
lookup_frames <- list()
for (par in params) {
  for (sx in sexes) {
    sub <- centiles_all[centiles_all$param == par$label & centiles_all$sex == sx, ]
    for (a in age_lookup) {
      closest_idx <- which.min(abs(sub$age_yrs[sub$centile == cents[1]] - a))
      for (cc in cents) {
        row <- sub[sub$centile == cc, ]
        lookup_frames[[length(lookup_frames) + 1]] <- data.frame(
          param = par$label, sex = sx, age_yrs = a, centile = cc,
          value = round(row$value[closest_idx], 2),
          stringsAsFactors = FALSE
        )
      }
    }
  }
}
lookup <- do.call(rbind, lookup_frames)
tables_dir <- file.path(onedrive, "tables")
dir.create(tables_dir, showWarnings = FALSE, recursive = TRUE)
write.csv(lookup, file.path(tables_dir, "T_normative_lookup.tsv"), row.names = FALSE)

# Summary
sink(file.path(out_dir, "summary.md"))
cat("# Paper 2 — Analysis 1: GAMLSS normative centile curves\n\n")
cat(sprintf("N = %d observations, %d unique participants across %s.\n\n",
            nrow(df), length(unique(df$participant_id)),
            paste(waves, collapse = ", ")))
cat("## Distribution selection (GAIC with k=log(N))\n\n")
cat("| Param | Sex | Family | GAIC |\n|---|---|---|---|\n")
best_rows <- dist_sel[order(dist_sel$param, dist_sel$sex, dist_sel$gaic), ]
for (i in seq_len(nrow(best_rows))) {
  r <- best_rows[i, ]
  cat(sprintf("| %s | %s | %s | %.1f |\n", r$param, r$sex, r$family, r$gaic))
}
cat("\n## Centile ranges at age 12 and 16 (P50)\n\n")
for (par in params) {
  for (sx in sexes) {
    sub12 <- lookup[lookup$param == par$label & lookup$sex == sx &
                    lookup$age_yrs == 12 & lookup$centile == 50, ]
    sub16 <- lookup[lookup$param == par$label & lookup$sex == sx &
                    lookup$age_yrs == 16 & lookup$centile == 50, ]
    if (nrow(sub12) > 0 && nrow(sub16) > 0) {
      cat(sprintf("- %s %s: P50 at age 12 = %.1f, age 16 = %.1f\n",
                  par$label, sx, sub12$value[1], sub16$value[1]))
    }
  }
}
sink()

cat("\nDone. Outputs in:", out_dir, "\n")
