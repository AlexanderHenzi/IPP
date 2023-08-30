# --- global settings ----------------------------------------------------------

# packages
library(tidyverse)
library(scales)

# ggplot2 settings, colors
theme_set(theme_bw(base_size = 12))
colpal <- c(
  "#999999",
  "#E69F00",
  "#56B4E9",
  "#009E73",
  "#0072B2",
  "#D55E00",
  "#CC79A7",
  "#F0E442"
)

# --- get data -----------------------------------------------------------------
fl <- paste0("temporary_files/simulation_results_logs/simulations_", seq_len(7000), ".rda")

data <- selection_data <- vector("list", length(fl))
pb <- txtProgressBar(max = length(fl))
for (i in seq_along(fl)) {
  setTxtProgressBar(pb, i)
  load(fl[i])
  data[[i]] <- tibble(
    n_obs_env = n_obs_env,
    lambda = seq(0, 15, 0.5),
    sim = sim,
    high_variance = high_variance_error,
    low_variance = low_variance_error,
    mean_shift = mean_shift_error,
    observational = prediction_error,
    correlation = corr_error,
    mean_err = mean_err,
    var_err = var_err,
    beta = list(beta),
    gamma = list(gamma),
    betahat = asplit(location, 2),
    gammahat = asplit(scale, 2)
  )
  selection_data[[i]] <- tibble(
    n_obs_env = n_obs_env,
    alpha_choose = alpha_choose,
    chosen_lambda = chosen_lambda,
    beta = list(beta),
    gamma = list(gamma),
    betahat = asplit(location[, match(chosen_lambda, seq(0, 15, 0.5))], 2),
    gammahat = asplit(scale[, match(chosen_lambda, seq(0, 15, 0.5))], 2)
  )
}
close(pb)
data <- do.call(rbind, data)
selection_data <- do.call(rbind, selection_data)

# --- plots --------------------------------------------------------------------
score <- "LogS"
alpha <- 0.05
load(paste0("data/simulation_results_", str_to_lower(score), ".rda"))

interventions <- data %>%
  gather(
    key = "intervention",
    value = "err",
    high_variance,
    low_variance,
    mean_shift,
    correlation,
    observational
  ) %>%
  mutate(
    intervention = factor(
      intervention,
      levels = c(
        "observational",
        "low_variance",
        "high_variance",
        "correlation",
        "mean_shift"
      ),
      labels = c(
        "pooled training",
        "low variance",
        "high variance",
        "correlation",
        "mean shift"
      ),
      ordered = TRUE
    )
  ) %>%
  filter(n_obs_env > 50) %>%
  group_by(n_obs_env, intervention, lambda) %>%
  summarise(err = mean(err)) %>%
  ggplot() +
  geom_line(
    aes(x = lambda, y = err, color = intervention, group = intervention)
  ) +
  scale_color_manual(values = colpal[seq_len(6)]) +
  facet_grid(cols = vars(n_obs_env)) +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(nrow = 1)) +
  labs(
    x = "Penalty",
    y = paste0("Test environment ", score),
    color = element_blank()
  )

if (score == "LogS") interventions <- 
  interventions + coord_cartesian(ylim = c(1.25, 3.5), xlim = c(0, 15))

pdf(
  paste0(
    "temporary_files/interventions_",
    str_to_lower(score),
    ".pdf"
  ),
  width = 8,
  height = 3
)
print(interventions)
dev.off()

parameter_error_data <- data %>%
  group_by(n_obs_env, lambda) %>%
  nest() %>%
  mutate(
    beta_bias = map(
      .x = data,
      .f = ~colMeans((do.call(rbind, .$betahat) - do.call(rbind, .$beta)))
    ),
    gamma_bias = map(
      .x = data,
      .f = ~colMeans((do.call(rbind, .$gammahat) - do.call(rbind, .$gamma)))
    ),
    beta_sqbias = map_dbl(beta_bias, ~sum(.^2)),
    gamma_sqbias = map_dbl(gamma_bias, ~sum(.^2)),
    beta_var = map2_dbl(
      .x = beta_bias,
      .y = data,
      ~sum(rowMeans((do.call(cbind, .y$betahat) - do.call(cbind, .y$beta) - .x)^2))
    ),
    gamma_var = map2_dbl(
      .x = gamma_bias,
      .y = data,
      ~sum(rowMeans((do.call(cbind, .y$gammahat) - do.call(cbind, .y$gamma) - .x)^2))
    ),
    beta_err = beta_sqbias + beta_var,
    gamma_err = gamma_sqbias + gamma_var,
    beta_se = map_dbl(
      .x = data,
      ~sd(rowSums((do.call(rbind, .x$betahat) - do.call(rbind, .x$beta))^2)) / 
        sqrt(nrow(.x))
    ),
    gamma_se = map_dbl(
      .x = data,
      ~sd(rowSums((do.call(rbind, .x$gammahat) - do.call(rbind, .x$gamma))^2)) / 
        sqrt(nrow(.x))
    ),
    beta_pse = beta_err + beta_se,
    beta_mse = beta_err - beta_se,
    gamma_pse = gamma_err + gamma_se,
    gamma_mse = gamma_err - gamma_se
  ) %>%
  select(-beta_se, -gamma_se) %>%
  pivot_longer(
    cols = c(
      "beta_err",
      "gamma_err",
      "beta_sqbias",
      "gamma_sqbias",
      "beta_var",
      "gamma_var",
      "beta_pse",
      "beta_mse",
      "gamma_pse",
      "gamma_mse"
    ),
    names_to = c("parameter", "error_measure"),
    names_sep = "_",
    values_to = "error"
  ) %>%
  mutate(
    error_measure = factor(
      error_measure,
      levels = c("err", "sqbias", "var", "pse", "mse"),
      labels = c("squared~L[2]~norm", "squared~bias", "variance", "pse", "mse"),
      ordered = TRUE
    )
  )

lambda_data <- selection_data %>%
  filter(n_obs_env > 50 & alpha_choose == alpha) %>%
  select(-gamma, -betahat, -gammahat, -beta) %>%
  group_by(n_obs_env, alpha_choose) %>%
  nest() %>%
  mutate(
    data = map(data, ~tally(group_by(., chosen_lambda)))
  ) %>%
  unnest(cols = data) %>%
  mutate(n = n / sum(n), alpha_choose = factor(alpha_choose))
lambda_data_beta <- lambda_data %>%
  group_by(n_obs_env) %>%
  mutate(n = n / max(n) * 1e-1, parameter = "beta")
lambda_data_gamma <- lambda_data %>%
  group_by(n_obs_env) %>%
  mutate(n = n / max(n) * 0.015, parameter = "gamma")
lambda_data <- rbind(lambda_data_beta, lambda_data_gamma)

parameter_error <- ggplot() +
  geom_linerange(
    data = lambda_data,
    aes(x = chosen_lambda, ymin = 0, ymax = n),
    alpha = 0.3,
    color = colpal[7],
    lwd = 1
  ) +
  geom_line(
    data = filter(
      parameter_error_data,
      n_obs_env > 50 & !(error_measure %in% c("pse", "mse"))
    ),
    aes(
      x = lambda,
      y = error,
      linetype = error_measure,
      color = error_measure
    )
  ) +
  scale_color_manual(values = colpal[1:3], labels = scales::parse_format()) +
  scale_fill_manual(values = colpal[7], labels = scales::parse_format()) +
  scale_linetype_manual(values = c(1, 5, 4), labels = scales::parse_format()) +
  facet_grid(
    rows = vars(parameter), cols = vars(n_obs_env),
    scales = "free_y",
    labeller = labeller(parameter = label_parsed)
  ) +
  theme(legend.position = "bottom") +
  labs(
    x = "Penalty",
    y = "Error",
    color = element_blank(),
    linetype = element_blank()
  )

pdf(
  paste0(
    "temporary_files/parameter_error_",
    str_to_lower(score),
    ".pdf"
  ),
  width = 8,
  height = 4
)
print(parameter_error)
dev.off()
