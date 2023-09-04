# --- setup --------------------------------------------------------------------
# packages
library(scoringRules)
library(tidyverse)
library(scales)

# ggplot2 settings, colors
theme_set(theme_bw(base_size = 12))
colpal <- c(
  "#999999",
  "#E69F00",
  "#56B4E9",
  "#009E73",
  "#F0E442",
  "#0072B2",
  "#D55E00",
  "#CC79A7"
)

# --- parameters ---------------------------------------------------------------
nsim <- 99999

# --- scores -------------------------------------------------------------------
logs_gaussian <- function(y, mean, sd) {
  -dnorm(y, mean, sd, log = TRUE)
}

crps_gaussian <- function(y, mean, sd) {
  crps_norm(y, mean, sd)
}

qs_gaussian <- function(y, mean, sd) {
  0.5 / sd / sqrt(2) - 2 * dnorm(y, mean, sd)
}

scrps_gaussian <- function(y, mean, sd) {
  z <- (mean - y) / sd
  sqrt(pi) * dnorm(z) + sqrt(pi) * z * 
    (2 * pnorm(z) - 1) / 2 + log(2 * sd / sqrt(pi)) / 2
}

pseudos_gaussian <- function(y, mean, sd, a = 2) {
  -dnorm(y, mean, sd)^(a - 1) * sd^((1 - a)^2/a) * (2*pi)^((1-a)^2/2/a) * a^(-(1/a - 1)/2)
}

hyvs_gaussian <- function(y, mean, sd) {
  (y - mean)^2 / sd^4 - 2/sd^2
}

compute_score <- function(y, mean, sd, score) {
  out <- switch(
    score,
    LogS = logs_gaussian(y, mean, sd),
    CRPS = crps_gaussian(y, mean, sd),
    SCRPS = scrps_gaussian(y, mean, sd),
    QS = qs_gaussian(y, mean, sd),
    PseudoS = pseudos_gaussian(y, mean, sd),
    Hyvarinen = hyvs_gaussian(y, mean, sd)
  )
  mean(out)
}

# --- simulate data ------------------------------------------------------------
set.seed(19910804)
eps_x_1 <- rnorm(nsim)
eps_x_2 <- rnorm(nsim)
r <- rnorm(nsim)
eps_y <- eps_x_1 + eps_x_2 + rnorm(nsim)

n_shift <- 200
shift <- seq(0, 1 - 1 / n_shift, 1 / n_shift)

scores_cond <- scores_ipp <- scores_cond_do <- scores_ipp_do <- data.frame(
  score = c("LogS", "CRPS", "SCRPS", "PseudoS", "QS", "Hyvarinen")
)

for (i in seq_along(shift)) {
  scores_cond <- cbind(scores_cond, new_col = 0)
  scores_ipp <- cbind(scores_ipp, new_col = 0)
  scores_cond_do <- cbind(scores_cond_do, new_col = 0)
  scores_ipp_do <- cbind(scores_ipp_do, new_col = 0)
  colnames(scores_cond)[i + 1] <- colnames(scores_ipp)[i + 1] <- 
    colnames(scores_cond_do)[i + 1] <- colnames(scores_ipp_do)[i + 1] <-
    paste0("shift_", shift[i])
  x_1 <- eps_x_1 + shift[i] * eps_x_2
  x_2 <- eps_x_2 + shift[i] * eps_x_1
  y <- x_1 + exp(x_2) * eps_y
  y_do <- shift[i] + exp(-1.5 + shift[i]^2) * eps_y
  lc <- x_1 + exp(x_2) * (x_1 + x_2)
  sc <- exp(x_2)
  lipp <- x_1
  sipp <- sqrt(3) * exp(x_2)
  for (j in seq_along(scores_cond$score)) {
    scores_cond[j, ncol(scores_cond)] <- 
      compute_score(y, lc, sc, scores_cond$score[j])
    scores_ipp[j, ncol(scores_ipp)] <- 
      compute_score(y, lipp, sipp, scores_ipp$score[j])
    scores_cond_do[j, ncol(scores_cond_do)] <- 
      compute_score(y_do, lc, sc, scores_ipp$score[j])
    scores_ipp_do[j, ncol(scores_ipp_do)] <- 
      compute_score(y_do, lipp, sipp, scores_ipp$score[j])
  }
}
scores_cond$method <- scores_cond_do$method <- "Conditional distribution"
scores_ipp$method <- scores_ipp_do$method <- "Do-interventional distribution"
scores_cond$interv <- scores_ipp$interv <- "X^t == Gamma[t]~epsilon[X]"
scores_cond_do$interv <- scores_ipp_do$interv <- "list(X[1]^t %==% t, X[2]^t %==% -1.5+t^2)"
scores <- rbind(scores_cond, scores_ipp, scores_cond_do, scores_ipp_do)

# --- plot ---------------------------------------------------------------------
example <- scores %>%
  mutate(
    score = factor(
      score,
      levels = c("LogS", "CRPS", "SCRPS", "PseudoS", "QS", "Hyvarinen"),
      labels = c("LogS", "CRPS", "SCRPS", "PseudoS", "QS", "Hyvärinen"),
      ordered = TRUE
    ),
    method = factor(
      method,
      levels = c("Do-interventional distribution", "Conditional distribution"),
      labels = c("Do-interventional distribution", "Conditional distribution"),
      ordered = TRUE
    ),
    interv = factor(
      interv,
      levels = c(
        "X^t == Gamma[t]~epsilon[X]",
        "list(X[1]^t %==% t, X[2]^t %==% -1.5+t^2)"
      ),
      labels = c(
        "X^t == Gamma[t]~epsilon[X]",
        "list(X[1]^t %==% t, X[2]^t %==% -1.5+t^2)"
      ),
      ordered = TRUE
    )
  ) %>%
  pivot_longer(
    cols = contains("shift"),
    names_to = "shift",
    values_to = "value"
  ) %>%
  mutate(
    shift = parse_number(shift),
    value = ifelse(value > 8 & score == "LogS", NA, value),
    value = ifelse(value > 3000 & score == "Hyvärinen", NA, value),
    value = ifelse(value > 3 & score == "SCRPS", NA, value)
  ) %>%
  ggplot() +
  geom_line(aes(
    x = shift,
    y = value,
    color = method,
    group = interaction(method, interv),
    linetype = interv
  )) +
  scale_linetype_manual(values = c(1, 5), labels = parse_format()) +
  facet_wrap(.~score, nrow = 2, scales = "free") +
  theme(legend.position = "bottom") +
  scale_color_manual(values = colpal[1:2]) +
  labs(
    x = expression(Intervention~strength~italic(t)),
    y = "Score",
    color = element_blank(),
    linetype = element_blank()
  )

pdf("temporary_files/illustrative_example.pdf", width = 8, height = 4)
print(example)
dev.off()

