# --- global settings ----------------------------------------------------------

# packages

## general and plotting
library(ggplot2)
library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(ggpubr)

## for distributional anchor regression
library(anchor)
library(tram)
library(CVXR)

## for IPP
library(GA)

## for selecting environments for validation
library(energy)

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

# functions
source("ipp.R")
source("distributional_anchor_regression.R")

# seed
set.seed(20230531)

# --- data and preprocessing ---------------------------------------------------

# data and preprocessing
data <- read.table("dataset_rpe1_99.csv", header = TRUE, sep = ",")
vnames <- c("non-targeting", colnames(data)[grepl("ENSG", colnames(data))])
data$hidden <- !(data$interventions %in% vnames)
rvar <- "ENSG00000173812"

# illustrate the distributions under interventions
data %>%
  select(-hidden) %>%
  filter(interventions %in% vnames) %>%
  gather(key = "gene", value = "expression", -interventions) %>%
  mutate(
    type = (interventions == gene) + 2 * (interventions == "non-targeting"),
    type = factor(
      type,
      levels = 0:2,
      labels = c(
        "other gene",
        "same gene",
        "observational"
      )
    ),
    ordered = TRUE
  ) %>%
  ggplot() +
  geom_density(aes(x = expression, group = interventions, color = type)) +
  scale_color_manual(values = colpal[c(1, 7, 4)]) +
  facet_wrap(. ~ gene, nrow = 2) +
  theme(legend.position = "bottom") +
  labs(x = "Gene expression", y = "Density", col = element_blank())

# create training and validation data
X_tr <- data %>%
  filter(!hidden) %>%
  select(-contains(rvar), -hidden) %>%
  group_by(interventions) %>%
  nest()
envs_interv <- X_tr$interventions
X_tr <- X_tr %>%
  getElement(name = "data") %>%
  map(.f = data.matrix)
Y_tr <- data %>%
  filter(!hidden) %>%
  select(interventions, contains(rvar)) %>%
  group_by(interventions) %>%
  nest() %>%
  getElement(name = "data") %>%
  map(.f = ~getElement(., rvar))

# energy distance to select environments for prediction
all_envs <- unique(data$interventions[data$hidden])
all_envs <- all_envs[all_envs != "excluded"]
obs_data <- data %>%
  filter(interventions == "non-targeting") %>%
  select(-hidden, -interventions) %>%
  data.matrix()
obs_data <- do.call(
  rbind,
  mapply(
    function(x, y) cbind(x, y),
    x = X_tr,
    y = Y_tr,
    SIMPLIFY = FALSE
  )
)
obs_data <- data.matrix(obs_data)
nobs <- nrow(obs_data)

m <- length(all_envs)
edists <- numeric(m)
pb <- txtProgressBar(max = m)
for (j in seq_len(m)) {
  intv_data <- data %>%
    filter(interventions == all_envs[j]) %>%
    select(-hidden, -interventions) %>%
    data.matrix()
  edists[j] <- eqdist.e(rbind(obs_data, intv_data), sizes = c(nobs, nrow(intv_data)))
  setTxtProgressBar(pb, j)
}
close(pb)
nenv <- 50
envs <- all_envs[order(edists, decreasing = TRUE)][seq_len(nenv)]

X_vl <- data %>%
  filter(hidden & interventions %in% envs)
envs_long <- X_vl$interventions
X_vl <- X_vl %>%
  select(-interventions, -contains(rvar), -hidden) %>%
  data.matrix()
Y_vl <- data %>%
  filter(hidden & interventions %in% envs) %>%
  select(contains(rvar)) %>%
  getElement(name = rvar)

# --- compare methods ----------------------------------------------------------

# fit different models to the training data

## our method

### scores
logs_gaussian <- function(x, l, s) -dnorm(x, mean = l, sd = s, log = TRUE)
scrps_gaussian <- function(y, l, s) {
  z <- (l - y) / s
  out <- sqrt(pi) * dnorm(z) + sqrt(pi) * z * 
    (2 * pnorm(z) - 1) / 2 + log(2 * s / sqrt(pi)) / 2
  out
}

### fit the model with logarithmic score and scrps
lambda <- c(0:5, seq(10, 50, 5))
nlambda <- length(lambda)

npar <- 2 * ncol(X_tr[[1]]) + 2
start_pop <- matrix(nrow = 200, ncol = npar, runif(npar * 200, -5, 5)) 
population <- function(object, ...) start_pop

fit_logs <- ipp(
  X = lapply(X_tr[envs_interv != rvar], function(x) cbind(1, x)),
  Y = Y_tr[envs_interv != rvar],
  lambda = lambda,
  score = logs_gaussian,
  type = "real-valued",
  nBits = NULL,
  lower = rep(-5, npar),
  upper = rep(5, npar),
  optim = TRUE,
  optimArgs = list(method = "CG", poptim = 0.1, maxit = 200),
  monitor = FALSE,
  popSize = 200,
  maxiter = 500,
  population = population,
  seed = 20230531
)

fit_scrps <- ipp(
  X = lapply(X_tr[envs_interv != rvar], function(x) cbind(1, x)),
  Y = Y_tr[envs_interv != rvar],
  lambda = lambda,
  score = scrps_gaussian,
  type = "real-valued",
  nBits = NULL,
  lower = rep(-5, npar),
  upper = rep(5, npar),
  optim = TRUE,
  optimArgs = list(method = "CG", poptim = 0.1, maxit = 200),
  monitor = FALSE,
  popSize = 200,
  maxiter = 500,
  population = population,
  seed = 20230531
)

### find out which lambda would be a good choice
pvals_logs <- pvals_scrps <- numeric(nlambda)
for (j in seq_len(nlambda)) {
  pvals_logs[j] <- oneway.test(
    err ~ env,
    data = do.call(
      rbind,
      imap(
        .x = fit_logs$errors,
        ~data.frame(env = .y, err = .x[, j])
      )
    )
  )$p.value
  pvals_scrps[j] <- oneway.test(
    err ~ env,
    data = do.call(
      rbind,
      imap(
        .x = fit_scrps$errors,
        ~data.frame(env = .y, err = .x[, j])
      )
    )
  )$p.value
}

plt_pval <- ggplot() +
  geom_hline(yintercept = c(0.05, 0.1), col = colpal[1], lty = 5) +
  geom_line(
    data = data.frame(
      penalty = lambda,
      pvalue = c(pvals_logs, pvals_scrps),
      score = rep(c("LogS", "SCRPS"), each = length(pvals_logs))
    ),
    aes(x = penalty, y = pvalue, color = score, group = score)
  ) +
  scale_color_manual(values = colpal[1:2]) +
  labs(x = "Penalty", y = "P-value", color = element_blank()) +
  theme(legend.position = "bottom")
plt_pval <- ggarrange(
  ggplot() + theme_void(),
  plt_pval,
  ggplot() + theme_void(),
  nrow = 1
)

pdf("temporary_files/single_cell_pval.pdf", width = 8, height = 2.5)
print(plt_pval)
dev.off()

### plot parameters as function of lambda
ipp_locs <- data.frame(
  value = c(c(fit_logs$location), c(fit_scrps$location)),
  lambda = rep(lambda, each = nrow(fit_logs$location)),
  variable = c("intercept", paste0("X", seq_len(nrow(fit_logs$location) - 1))),
  type = "location",
  score = rep(c("logs", "scrps"), each = length(fit_logs$location))
)
ipp_scales <- data.frame(
  value = c(c(fit_logs$scale), c(fit_scrps$scale)),
  lambda = rep(lambda, each = nrow(fit_logs$location)),
  variable = c("intercept", paste0("X", seq_len(nrow(fit_logs$location) - 1))),
  type = "scale",
  score = rep(c("logs", "scrps"), each = length(fit_logs$location))
)
ipp_par <- rbind(ipp_locs, ipp_scales)

ggplot() +
  geom_line(
    data = ipp_par,
    aes(
      x = lambda,
      y = value,
      color = variable,
      group = interaction(variable, score),
      linetype = score
    )
  ) +
  facet_wrap(.~type) +
  theme(legend.position = "bottom")

### plot errors as function of lambda and environment
ipp_logs_errs <- imap(
  .x = fit_logs$errors[order(lengths(fit_logs$errors))],
  .f = ~data.frame(env = .y, logs = c(t(.x)), lambda = lambda, score = "LogS")
)
ipp_scrps_errs <- imap(
  .x = fit_scrps$errors[order(lengths(fit_scrps$errors))],
  .f = ~data.frame(env = .y, logs = c(t(.x)), lambda = lambda, score = "SCRPS")
)
ipp_errs <- do.call(rbind, c(ipp_logs_errs, ipp_scrps_errs)) %>%
  filter(lambda %in% c(0, 5, 10, 20, 40))

stability <- ggplot() +
  geom_errorbar(
    data = summarise(
      group_by(ipp_errs, env, lambda),
      lwr = quantile(logs, 0.05),
      upr = quantile(logs, 0.95)
    ),
    aes(x = env, ymin = lwr, ymax = upr),
    col = "darkgray"
  ) +
  geom_line(
    data = summarise(group_by(ipp_errs, env, lambda, score), logs = mean(logs)),
    aes(x = as.numeric(as.factor(env)), y = logs)
  ) +
  geom_point(
    data = summarise(group_by(ipp_errs, env, lambda, score), logs = mean(logs)),
    aes(x = env, y = logs),
    pch = 19
  ) +
  geom_errorbar(
    data = summarise(
      group_by(ipp_errs, env, lambda, score),
      lwr = quantile(logs, 0.25),
      upr = quantile(logs, 0.75)
    ),
    aes(x = env, ymin = lwr, ymax = upr)
  ) +
  facet_grid(cols = vars(lambda), rows = vars(score)) +
  labs(x = "Environments", y = "LogS") +
  theme(
    axis.ticks.x = element_blank(),
    axis.text.x = element_blank()
  )

pdf("temporary_files/stability.pdf", width = 8, height = 2.5)
print(stability)
dev.off()

### get predicted location and scale parameter
pred_logs <- predict(fit_logs, cbind(1, X_vl))
pred_scrps <- predict(fit_scrps, cbind(1, X_vl))

### compute log score, squared error, sscrps
logs_logs <- logs_gaussian(Y_vl, pred_logs$location, pred_logs$scale)
logs_scrps <- scrps_gaussian(Y_vl, pred_logs$location, pred_logs$scale)
logs_sqerr <- (pred_logs$location - Y_vl)^2
scrps_logs <- logs_gaussian(Y_vl, pred_scrps$location, pred_scrps$scale)
scrps_scrps <- scrps_gaussian(Y_vl, pred_scrps$location, pred_scrps$scale)
scrps_sqerr <- (pred_logs$location - Y_vl)^2

## distributional anchor regression
xi <- lambda
o_env <- which(envs_interv == "non-targeting")
dar_fit <- dar_c_probit(
  X_tr[-o_env],
  Y_tr[-o_env],
  xi = xi,
  support = c(0.5, 5.5)
)

### compute log score, squared error, scrps
q <- seq(0.5, 5.5, 0.05)
q[1] <- q[1] + 1e-5
q[length(q)] <- q[length(q)] - 1e-5
dar_logs <- logs_dar(dar_fit, X_vl, Y_vl)
dar_scrps <- scrps_dar(dar_fit, X_vl, Y_vl, q)
dar_sqerr <- sqerr_dar(dar_fit, X_vl, Y_vl, q)

## compute mean scores for each environment
err_data <- list(
  logs_logs,
  logs_scrps,
  logs_sqerr,
  scrps_logs,
  scrps_scrps,
  scrps_sqerr,
  dar_logs,
  dar_scrps,
  dar_sqerr
)
methods <- rep(
  factor(
    c("IPP (LogS)", "IPP (SCRPS)", "Distributional anchor"),
    levels = c(
      "IPP (LogS)",
      "IPP (SCRPS)",
      "Distributional anchor",
      "DRIG",
      "Anchor regression"
    ),
    labels = c(
      "IPP (LogS)",
      "IPP (SCRPS)",
      "Distributional anchor",
      "DRIG",
      "Anchor regression"
    ),
    ordered = TRUE
  ),
  each = 3
)
scores <- rep(factor(
  c("Logs", "SCRPS", "MSE"),
  levels =  c("Logs", "SCRPS", "MSE"),
  labels =  c("Logs", "SCRPS", "MSE"),
  ordered = TRUE),
  3
)
df <- mapply(
  function(data, method, score) {
    colnames(data) <- paste0("lambda_", lambda)
    data <- as_tibble(data)
    data$intv <- envs_long
    data %>% 
      group_by(intv) %>%
      summarise_all(mean) %>%
      gather(key = "penalty", value = "err", -intv) %>%
      mutate(penalty = parse_number(penalty), method = method, score = score)
  },
  data = err_data,
  method = methods,
  score = scores,
  SIMPLIFY = FALSE
)
df <- do.call(rbind, df)

## get results for anchor regression and drig
an_dr_data <- 
  read.csv("temporary_files/anchor_drig.csv", header = TRUE, sep = ",")
an_dr_data <- an_dr_data %>%
  filter(gamma %in% lambda & interv_gene %in% envs) %>%
  mutate(
    method = ifelse(
      method == "anchor regression",
      "Anchor regression",
      "DRIG"
    ),
    method = factor(
      method,
      levels = levels(df$method),
      labels = levels(df$method),
      ordered = TRUE
    )
  ) %>%
  mutate(score = factor(
    "MSE",
    levels = c("Logs", "SCRPS", "MSE"),
    labels = c("Logs", "SCRPS", "MSE"),
    ordered = TRUE
    )
  ) %>%
  select(-X) %>%
  rename(penalty = gamma, err = test_mse, intv = interv_gene) %>%
  filter(penalty %in% df$penalty)
df <- rbind(df, an_dr_data)

## quantiles of the scores
qs <- seq(0, 1, 0.1)
df_qs <- df %>%
  group_by(method, penalty, score) %>%
  summarise(
    out = list(tibble(qs, err = quantile(err, probs = qs, type = 1)))
  ) %>%
  unnest(cols = out)

## plots
score_plots <- vector("list", 3)
for (i in seq_len(3)) {
  df_tmp <- df_qs %>%
    filter(score == levels(scores)[i])

  plt <- ggplot() +
    geom_line(
      data = filter(df_tmp, abs(qs - 0.5) < 1e-5),
      aes(x = penalty, y = err)
    )
  for (j in ((length(qs)- 1) / 2 - 1):1) {
    q <- qs[j]
    df_tmp1 <- filter(df_tmp, abs(qs - q) < 1e-5) %>%
      rename(y_lwr = err)
    df_tmp2 <- filter(df_tmp, abs(qs - 1 + q) < 1e-5) %>%
      rename(y_upr = err)
    plt <- plt +
      geom_ribbon(
        data = cbind(df_tmp1, y_upr = df_tmp2$y_upr),
        aes(x = penalty, ymin = y_lwr, ymax = y_upr),
        alpha = 0.15
      )
  }
  mean_errors <- df_tmp %>%
    filter(score == levels(scores)[i]) %>%
    group_by(method, score, penalty) %>%
    summarise(err = mean(err))
  
  plt <- plt +
    geom_line(
      data = mean_errors,
      aes(x = penalty, y = err),
      lty = 5
    ) +
    labs(
      x = if (i == 3) "Penalty" else element_blank(),
      y = levels(scores)[i],
      color = element_blank(),
      fill = element_blank()
    ) +
    facet_grid(cols = vars(method))
  if (i == 3) plt <- plt + coord_cartesian(ylim = c(0.1, 0.9))
  score_plots[[i]] <- plt
}

single_cell_test <- ggarrange(
  plotlist = score_plots,
  heights = c(1, 1, 1.2),
  ncol = 1
)

pdf("temporary_files/single_cell_test.pdf", width = 8, height = 6)
print(single_cell_test)
dev.off()
