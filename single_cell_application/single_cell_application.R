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
source("functions/ipp.R")
source("functions/distributional_anchor_regression.R")

# seed
set.seed(20230531)

# whether to run computations
run_computations <- FALSE

#-------------------------------------------------------------------------------
# plot in main part of paper
if (!run_computations) {
  load("data/single_cell.rda")
}

# --- data and preprocessing ---------------------------------------------------

# data and preprocessing
data <- read.table("data/dataset_rpe1_99.csv", header = TRUE, sep = ",")
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
oe <- which.max(lengths(Y_tr))
training <- as.data.frame(cbind(X_tr[[oe]], Y_tr[[oe]]))
#write.table(training, "training.csv", sep = ";", row.names = FALSE)

# energy distance to select environments for prediction, compare to OLS
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

## for OLS prediction 
observational_means <- c(
  colMeans(X_tr[[which(envs_interv == "non-targeting")]]),
  mean(Y_tr[[which(envs_interv == "non-targeting")]])
)
obs_data_ols <- t(t(obs_data) - observational_means)
obs_x_ols <- obs_data_ols[, -ncol(obs_data_ols)]
obs_y_ols <- obs_data_ols[, ncol(obs_data_ols)]
clm <- coef(lm(obs_y_ols ~ obs_x_ols - 1))

## compute energy distance and errors
nenv <- 50
if (run_computations) {
  edists <- ols_mse <- numeric(m)
  pb <- txtProgressBar(max = m)
  for (j in seq_len(m)) {
    intv_data <- data %>%
      filter(interventions == all_envs[j]) %>%
      select(-hidden, -interventions) %>%
      data.matrix()
    ols_mse[j] <- mean(
      (intv_data[, ncol(intv_data)] - intv_data[, -ncol(intv_data)] %*% clm)^2
    )
    edists[j] <- eqdist.e(
      rbind(obs_data, intv_data),
      sizes = c(nobs, nrow(intv_data))
    )
    setTxtProgressBar(pb, j)
  }
  close(pb)
  envs <- all_envs[order(edists, decreasing = TRUE)][seq_len(nenv)]
  envs_ols <- all_envs[order(ols_mse, decreasing = TRUE)][seq_len(nenv)]
  intersect(envs, envs_ols)
  length(intersect(envs, envs_ols))
}

## format validation data
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
#test <- cbind(env = envs_long, as.data.frame(cbind(X_vl, Y_vl)))
#write.table(test, "test.csv", sep = ";", row.names = FALSE)

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
crps_gaussian <- function(x, l, s) 
  scoringRules::crps_norm(y = x, mean = l, sd = s)

### fit the model with logarithmic score and scrps
lambda <- c(0:5, seq(10, 70, 5))
nlambda <- length(lambda)

npar <- 2 * ncol(X_tr[[1]]) + 2
popSize <- 200
start_pop <- matrix(nrow = popSize, ncol = npar, runif(npar * popSize, -5, 5)) 
population <- function(object, ...) start_pop

if (run_computations) {
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
    popSize = popSize,
    maxiter = 800,
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
    popSize = popSize,
    maxiter = 800,
    population = population,
    seed = 20230531
  )
  set.seed(20230531)
  start_pop <- matrix(nrow = 400, ncol = npar, runif(npar * 400, -5, 5))
  fit_crps <- ipp(
    X = lapply(X_tr[envs_interv != rvar], function(x) cbind(1, x)),
    Y = Y_tr[envs_interv != rvar],
    lambda = lambda,
    score = crps_gaussian,
    type = "real-valued",
    nBits = NULL,
    lower = rep(-5, npar),
    upper = rep(5, npar),
    optim = TRUE,
    optimArgs = list(method = "CG", poptim = 0.1, maxit = 200),
    monitor = FALSE,
    popSize = 400,
    maxiter = 500,
    population = population,
    seed = 20230531
  )
  save(list = ls(), file = "data/single_cell.rda")
}

### find out which lambda would be a good choice
if (run_computations) {
  ipp_fits <- list(LogS = fit_logs, CRPS = fit_crps, SCRPS = fit_scrps)
}
ipp_preds <- lapply(
  ipp_fits,
  function(x) predict(x, cbind(1, X_vl))
)
pvals_data <- rep(list(numeric(nlambda)), 3)
names(pvals_data) <- names(ipp_fits)
for (i in seq_along(ipp_fits)) {
  for (j in seq_len(nlambda)) {
    pvals_data[[i]][j] <- oneway.test(
      err ~ env,
      data = do.call(
        rbind,
        imap(
          .x = ipp_fits[[i]]$errors,
          ~data.frame(env = .y, err = .x[, j])
        )
      )
    )$p.value
  }
}
pval_data <- data.frame(
  pvalue = unlist(pvals_data),
  score = rep(names(ipp_fits), each = nlambda),
  penalty = lambda
)
pval_data <- pval_data[pval_data$score != "CRPS", ]

plt_pval <- ggplot() +
  geom_hline(yintercept = c(0.05, 0.1), col = colpal[1], lty = 5) +
  geom_line(
    data = filter(pval_data, penalty <= 70),
    aes(x = penalty, y = pvalue, color = score, group = score)
  ) +
  scale_color_manual(values = colpal[1:3]) +
  labs(x = "Penalty", y = "P-value", color = element_blank()) +
  # theme(legend.position = "bottom") +
  theme(legend.position = "none") +
  coord_cartesian(xlim = c(0, 70))

### plot errors as function of lambda and environment
ipp_errs <- mapply(
  function(x, y) {
    do.call(rbind, imap(
      .x = x$errors[order(lengths(x$errors))],
      .f = ~data.frame(env = .y, err = c(t(.x)), lambda = lambda, score = y)
    ))
  },
  x = ipp_fits,
  y = names(ipp_fits),
  SIMPLIFY = FALSE
)
ipp_errs <- do.call(rbind, ipp_errs)
ipp_errs <- ipp_errs %>%
  filter(lambda %in% c(0, 5, 10, 35, 70) & score != "CRPS")

stability <- ggplot() +
  geom_errorbar(
    data = summarise(
      group_by(ipp_errs, env, lambda),
      lwr = quantile(err, 0.05),
      upr = quantile(err, 0.95)
    ),
    aes(x = env, ymin = lwr, ymax = upr),
    col = "darkgray"
  ) +
  geom_line(
    data = summarise(group_by(ipp_errs, env, lambda, score), err = mean(err)),
    aes(x = as.numeric(as.factor(env)), y = err)
  ) +
  geom_point(
    data = summarise(group_by(ipp_errs, env, lambda, score), err = mean(err)),
    aes(x = env, y = err),
    pch = 19
  ) +
  geom_errorbar(
    data = summarise(
      group_by(ipp_errs, env, lambda, score),
      lwr = quantile(err, 0.25),
      upr = quantile(err, 0.75)
    ),
    aes(x = env, ymin = lwr, ymax = upr)
  ) +
  facet_grid(cols = vars(lambda), rows = vars(score)) +
  labs(x = "Environments", y = "Risk") +
  theme(
    axis.ticks.x = element_blank(),
    axis.text.x = element_blank()
  )

stability_pvals <- ggarrange(
  stability,
  plt_pval,
  nrow = 1,
  widths = c(2, 1)
)

pdf("temporary_files/stabilityPvals.pdf", width = 8, height = 2.5)
print(stability_pvals)
dev.off()

## distributional anchor regression
xi <- lambda
q <- seq(0.5, 5.5, 0.01)
q[1] <- q[1] + 1e-5
q[length(q)] <- q[length(q)] - 1e-5
if (run_computations) {
  dar_fit <- dar_c_probit(
    X_tr[!(envs_interv %in% c(rvar, "non-targeting"))],
    Y_tr[!(envs_interv %in% c(rvar, "non-targeting"))],
    xi = xi,
    support = c(0.5, 5.5)
  )
  dar_logs <- logs_dar(dar_fit, X_vl, Y_vl)
  dar_scrps <- scrps_dar(dar_fit, X_vl, Y_vl, q)
  dar_sqerr <- sqerr_dar(dar_fit, X_vl, Y_vl, q)
  dar_crps <- crps_dar(dar_fit, X_vl, Y_vl, q)
  dar_prdint <- predint_dar(dar_fit, X_vl, Y_vl)
  dar_in_pi <- dar_prdint[[1]]
  dar_pi_width <- dar_prdint[[2]]
}
dar_scores <- list(
  logs = dar_logs,
  crps = dar_crps,
  scrps = dar_scrps,
  sqerr = dar_sqerr,
  in_pi = dar_in_pi,
  pi_width = dar_pi_width
)

### compute and summarise log score, squared error, scrps
score_funs <- list(
  logs = function(pred) logs_gaussian(Y_vl, pred$location, pred$scale),
  crps = function(pred) crps_gaussian(Y_vl, pred$location, pred$scale),
  scrps = function(pred) scrps_gaussian(Y_vl, pred$location, pred$scale),
  sqerr = function(pred) (pred$location - Y_vl)^2,
  in_pi = function(pred) (Y_vl > qnorm(0.05, pred$location, pred$scale)) &
    (Y_vl < qnorm(0.95, pred$location, pred$scale)),
  pi_width = function(pred) qnorm(0.95, pred$location, pred$scale)- 
    qnorm(0.05, pred$location, pred$scale)
)
ipp_scores <- vector("list", length(ipp_fits) * length(score_funs))
k <- 1
for (j in seq_along(score_funs)) {
  err <- lapply(
    asplit(dar_scores[[j]], 2),
    function(x) tapply(x, envs_long, mean)
  )
  dar_scores[[j]] <- data.frame(
    env = names(unlist(err)),
    err = unname(unlist(err)),
    penalty = rep(lambda, each = length(err[[1]])),
    method = paste0("Distributional anchor"),
    score = names(score_funs)[j]
  )
  for (i in seq_along(ipp_fits)) {
    err <- lapply(
      asplit(score_funs[[j]](ipp_preds[[i]]), 2),
      function(x) tapply(x, envs_long, mean)
    )
    ipp_scores[[k]] <- data.frame(
      env = names(unlist(err)),
      err = unname(unlist(err)),
      penalty = rep(lambda, each = length(err[[1]])),
      method = paste0("IPP (", names(ipp_fits)[i], ")"),
      score = names(score_funs)[j]
    )
    k <- k + 1
  }
}
ipp_scores <- do.call(rbind, ipp_scores)
dar_scores <- do.call(rbind, dar_scores)

if (run_computations) {
  save(
    list = c("ipp_fits", "dar_logs", "dar_crps", "dar_scrps", "dar_sqerr",
             "dar_in_pi", "dar_pi_width", "envs"),
    file = "data/single_cell.rda"
  )
}

## get results for anchor regression and drig
an_dr_data <- 
  read.csv("data/anchor_drig.csv", header = TRUE, sep = ",")
an_dr_data <- an_dr_data %>%
  filter(gamma %in% lambda & interv_gene %in% envs) %>%
  select(-X) %>%
  rename(penalty = gamma, err = test_mse, env = interv_gene) %>%
  mutate(score = "sqerr")

## get results for neural networks
mse_rex <- read.csv("data/results_rex_mse.csv") %>%
  mutate(method = "vrex", score = "sqerr", env = envs[1]) %>%
  rename(err = test_mse, penalty = lambda)
mse_ipp_scrps_nn <- read.csv("data/results_ipp_mse.csv") %>%
  mutate(method = "ipp_scrps_nn", score = "sqerr", env = envs[1]) %>%
  rename(err = test_mse, penalty = lambda)
scrps_ipp_nn <- read.csv("data/results_ipp_scrps.csv")  %>%
  mutate(method = "ipp_scrps_nn", score = "scrps", env = envs[1]) %>%
  rename(err = test_scrps, penalty = lambda)
predint_ipp_scrps_nn <- read.csv("data/pred_interval_nn_scrps.csv") %>%
  mutate(method = "ipp_scrps_nn", env = envs[1]) %>%
  rename(in_pi = cover90, penalty = lambda, pi_width = length90) %>%
  select(-cover95, -length95) %>%
  gather(key = "score", value = "err", in_pi, pi_width)
predint_ipp_crps_nn <- read.csv("data/pred_interval_nn_crps.csv") %>%
  mutate(method = "ipp_crps_nn", env = envs[1]) %>%
  rename(in_pi = cover90, penalty = lambda, pi_width = length90) %>%
  select(-cover95, -length95) %>%
  gather(key = "score", value = "err", in_pi, pi_width)
crps_ipp_crps_nn <- 
  read.csv("data/results_crpsloss_crps_lr1e-3_epoch1000.csv") %>%
  mutate(method = "ipp_crps_nn", score = "crps", env = envs[1]) %>%
  rename(err = test_crps, penalty = lambda)
crps_ipp_scrps_nn <- 
  read.csv("data/results_scrpsloss_crps_lr1e-3_epoch1000.csv") %>%
  mutate(method = "ipp_scrps_nn", score = "crps", env = envs[1]) %>%
  rename(err = test_crps, penalty = lambda)
cp_files <- list.files("data/single_cell_cp/") 
cp_files <- paste0("data/single_cell_cp/", cp_files)

cp <- vector("list", length(cp_files))
for (i in seq_along(cp_files)) {
  cp[[i]] <- read.csv(cp_files[i], sep = ";")
}
cp <- do.call(rbind, cp)
colnames(cp)[colnames(cp) == "value"] <- "err"
cp$method <- paste0(
  cp$method,
  "_",
  ifelse(cp$training == "pooled", "pooled", "obs")
)
cp <- cp[, colnames(ipp_scores)]

## combine all results
df <- rbind(
  ipp_scores,
  dar_scores,
  mse_rex,
  mse_ipp_scrps_nn,
  an_dr_data,
  scrps_ipp_nn,
  predint_ipp_scrps_nn,
  predint_ipp_crps_nn,
  crps_ipp_crps_nn,
  crps_ipp_scrps_nn,
  cp
)
df <- df %>% 
  mutate(
    method = factor(
      method,
      labels = c(
        "IPP (LogS)",
        "IPP (SCRPS)",
        "IPP (CRPS)",
        "IPP (SCRPS, NN)",
        "IPP (CRPS, NN)",
        "Distributional anchor",
        "DRIG",
        "Anchor regression",
        "V-Rex",
        "Conformal (pool)",
        "Conformal (obs)",
        "Conformal (weighted, pool)",
        "Conformal (weighted, obs)"
      ),
      levels = c(
        "IPP (LogS)",
        "IPP (SCRPS)",
        "IPP (CRPS)",
        "ipp_scrps_nn",
        "ipp_crps_nn",
        "Distributional anchor",
        "DRIG",
        "anchor regression",
        "vrex",
        "rcp_pooled",
        "rcp_obs",
        "dwrcp_pooled",
        "dwrcp_obs"
      ),
      ordered = TRUE
    ),
    score = factor(
      score,
      levels = c("logs", "scrps", "crps", "sqerr", "in_pi", "pi_width"),
      labels = c(
        "LogS",
        "SCRPS",
        "CRPS",
        "MSE",
        "Coverage",
        "Length"
      ),
      ordered = TRUE
    )
  )

## for robust CP: avoid different lengths / coverages due to train test splits, average
df_rcp <- df %>%
  filter(method %in% c("Conformal (obs)", "Conformal (pool)")) %>%
  group_by(penalty, score, method) %>%
  mutate(err = mean(err))
df <- df %>%
  filter(!method %in% c("Conformal (obs)", "Conformal (pool)")) %>%
  bind_rows(df_rcp)

## quantiles of the scores
qs <- seq(0, 1, 0.1)
df_qs <- df %>%
  group_by(method, penalty, score) %>%
  summarise(
    out = list(tibble(qs, err = quantile(err, probs = qs, type = 1)))
  ) %>%
  unnest(cols = out)


## plots (paper)
df_qs_small <- df_qs %>%
  filter(score != "CRPS" & !grepl("(CRPS", method, fixed = TRUE) &
   !grepl("pool", method))
scores <- c("LogS", "SCRPS", "MSE", "MSE", "Coverage", "Length")
methods <- rep(list(unique(df_qs_small$method)), length(scores))
methods[[3]] <- c("IPP (LogS)", "IPP (SCRPS)", "IPP (SCRPS, NN)")
methods[[4]] <- c("Distributional anchor", "DRIG", "Anchor regression", "V-Rex")
ncols <- c(3, 4, 3, 4, 3, 3)
score_plots <- vector("list", length(scores))

for (i in seq_along(scores)) {
  df_tmp <- df_qs_small %>%
    filter(score == scores[i] & method %in% methods[[i]])
  plt <- ggplot() +
    geom_line(
      data = filter(df_tmp, abs(qs - 0.5) < 1e-5),
      aes(x = penalty, y = err)
    )
  if (scores[i] == "Coverage") {
    plt <- plt + 
      geom_hline(yintercept = 0.9, col = 2)
  }
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
  mean_errors <- df %>%
    filter(score == scores[i] & df$method %in% methods[[i]]) %>%
    group_by(method, score, penalty) %>%
    summarise(err = mean(err))
  
  plt <- plt +
    geom_line(
      data = mean_errors,
      aes(x = penalty, y = err),
      lty = 5
    ) +
    labs(
      x = if (i == length(scores)) "Penalty" else element_blank(),
      y = scores[i],
      color = element_blank(),
      fill = element_blank()
    ) +
    scale_y_continuous(labels = scales::number_format(accuracy = 0.01)) +
    facet_wrap(.~method, scales = "free_x", ncol = ncols[i]) +
    theme_bw(base_size = 10)
  score_plots[[i]] <- plt
}

heights <- c(1, 1, 1, 1, 2, 2.05)
single_cell_test <- ggarrange(
  plotlist = score_plots,
  heights = heights,
  ncol = 1
)

pdf("temporary_files/singleCellArticle.pdf", width = 8, height = 12)
print(single_cell_test)
dev.off()

## plots (supplement)
scores <- as.character(sort(unique(df_qs$score)))
ncols <- c(4, 5, 6, 4, 5, 5)
score_plots <- vector("list", length(scores))

for (i in seq_along(scores)) {
  df_tmp <- df_qs %>%
    filter(score == scores[i])
  plt <- ggplot() +
    geom_line(
      data = filter(df_tmp, abs(qs - 0.5) < 1e-5),
      aes(x = penalty, y = err)
    )
  if (scores[i] == "Coverage") {
    plt <- plt + 
      geom_hline(yintercept = 0.9, col = 2)
  }
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
  mean_errors <- df %>%
    filter(score == scores[i]) %>%
    group_by(method, score, penalty) %>%
    summarise(err = mean(err))
  
  plt <- plt +
    geom_line(
      data = mean_errors,
      aes(x = penalty, y = err),
      lty = 5
    ) +
    labs(
      x = if (i == length(scores)) "Penalty" else element_blank(),
      y = scores[i],
      color = element_blank(),
      fill = element_blank()
    ) +
    scale_y_continuous(labels = scales::number_format(accuracy = 0.01)) +
    facet_wrap(.~method, scales = "free_x", ncol = ncols[i]) +
    theme_bw(base_size = 10)
  score_plots[[i]] <- plt
}

heights <- c(1, 1, 1, 2, 2, 2.05)
single_cell_test <- ggarrange(
  plotlist = score_plots,
  heights = heights,
  ncol = 1
)

pdf("temporary_files/singleCellWithoutRvarAllMethods.pdf", width = 8, height = 12)
print(single_cell_test)
dev.off()


#-------------------------------------------------------------------------------
# plots including interventions on response variable for IPP

## clean environment
rm(list = ls())
run_computations <- FALSE
if (!run_computations) {
  load("data/single_cell_rvar.rda")
}

## functions
source("functions/ipp.R")
source("functions/distributional_anchor_regression.R")

# --- data and preprocessing ---------------------------------------------------

# data and preprocessing
data <- read.table("data/dataset_rpe1_99.csv", header = TRUE, sep = ",")
vnames <- c("non-targeting", colnames(data)[grepl("ENSG", colnames(data))])
data$hidden <- !(data$interventions %in% vnames)
rvar <- "ENSG00000173812"

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

# energy distance to select environments for prediction, compare to OLS
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

## compute energy distance and errors
nenv <- 50
if (run_computations) {
  edists <- numeric(m)
  pb <- txtProgressBar(max = m)
  for (j in seq_len(m)) {
    intv_data <- data %>%
      filter(interventions == all_envs[j]) %>%
      select(-hidden, -interventions) %>%
      data.matrix()
    edists[j] <- eqdist.e(
      rbind(obs_data, intv_data),
      sizes = c(nobs, nrow(intv_data))
    )
    setTxtProgressBar(pb, j)
  }
  close(pb)
  envs <- all_envs[order(edists, decreasing = TRUE)][seq_len(nenv)]
}

## format validation data
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
lambda <- c(0:5, seq(10, 70, 5))
nlambda <- length(lambda)

npar <- 2 * ncol(X_tr[[1]]) + 2
start_pop <- matrix(nrow = 200, ncol = npar, runif(npar * 200, -5, 5)) 
population <- function(object, ...) start_pop

if (run_computations) {
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
  save(list = ls(), file = "data/single_cell_rvar.rda")
}

fit_logs$lambda <- fit_logs$lambda <- lambda
fit_logs$location <- fit_logs$location[, seq_along(lambda)]
fit_logs$scale <- fit_logs$scale[, seq_along(lambda)]
fit_logs$risk <- fit_logs$risk[seq_along(lambda)]
fit_logs$diff <- fit_logs$diff[seq_along(lambda)]
fit_logs$errors <- lapply(
  fit_logs$errors,
  function(x) x[, seq_along(lambda)]
)
fit_scrps$lambda <- fit_scrps$lambda <- lambda
fit_scrps$location <- fit_scrps$location[, seq_along(lambda)]
fit_scrps$scale <- fit_scrps$scale[, seq_along(lambda)]
fit_scrps$risk <- fit_scrps$risk[seq_along(lambda)]
fit_scrps$diff <- fit_scrps$diff[seq_along(lambda)]
fit_scrps$errors <- lapply(
  fit_scrps$errors,
  function(x) x[, seq_along(lambda)]
)

### get predicted location and scale parameter
pred_logs <- predict(fit_logs, cbind(1, X_vl))
pred_scrps <- predict(fit_scrps, cbind(1, X_vl))

### compute log score, squared error, sscrps
logs_logs <- logs_gaussian(Y_vl, pred_logs$location, pred_logs$scale)
logs_scrps <- scrps_gaussian(Y_vl, pred_logs$location, pred_logs$scale)
logs_sqerr <- (pred_logs$location - Y_vl)^2
scrps_logs <- logs_gaussian(Y_vl, pred_scrps$location, pred_scrps$scale)
scrps_scrps <- scrps_gaussian(Y_vl, pred_scrps$location, pred_scrps$scale)
scrps_sqerr <- (pred_scrps$location - Y_vl)^2

## compute mean scores for each environment
err_data <- list(
  logs_logs,
  logs_scrps,
  logs_sqerr,
  scrps_logs,
  scrps_scrps,
  scrps_sqerr
)
methods <- rep(
  factor(
    c("IPP (LogS)", "IPP (SCRPS)"),
    levels = c(
      "IPP (LogS)",
      "IPP (SCRPS)"
    ),
    labels = c(
      "IPP (LogS)",
      "IPP (SCRPS)"
    ),
    ordered = TRUE
  ),
  each = 3
)
scores <- rep(factor(
  c("LogS", "SCRPS", "MSE"),
  levels =  c("LogS", "SCRPS", "MSE"),
  labels =  c("LogS", "SCRPS", "MSE"),
  ordered = TRUE),
  2
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
  method = rep(methods, each = 3),
  score = scores,
  SIMPLIFY = FALSE
)
df <- do.call(rbind, df)

## quantiles of the scores
qs <- seq(0, 1, 0.1)
df_qs <- df %>%
  group_by(method, penalty, score) %>%
  summarise(
    out = list(tibble(qs, err = quantile(err, probs = qs, type = 1)))
  ) %>%
  unnest(cols = out)

## plot
plt <- ggplot() +
  geom_line(
    data = filter(df_qs, abs(qs - 0.5) < 1e-5),
    aes(x = penalty, y = err)
  )
for (j in ((length(qs)- 1) / 2 - 1):1) {
  q <- qs[j]
  df_tmp1 <- filter(df_qs, abs(qs - q) < 1e-5) %>%
    rename(y_lwr = err)
  df_tmp2 <- filter(df_qs, abs(qs - 1 + q) < 1e-5) %>%
    rename(y_upr = err)
  plt <- plt +
    geom_ribbon(
      data = cbind(df_tmp1, y_upr = df_tmp2$y_upr),
      aes(x = penalty, ymin = y_lwr, ymax = y_upr),
      alpha = 0.15
    )
}
mean_errors <- df %>%
  group_by(method, score, penalty) %>%
  summarise(err = mean(err))

plt <- plt +
  geom_line(
    data = mean_errors,
    aes(x = penalty, y = err),
    lty = 5
  ) +
  labs(
    x = "Penalty",
    y = "Test error"
  ) +
  facet_grid(cols = vars(score), rows = vars(method))


pdf("temporary_files/singleCellRvar.pdf", width = 8, height = 4)
print(plt)
dev.off()

#-------------------------------------------------------------------------------
# plots excluding observational environment for IPP

## clean environment
rm(list = ls())
run_computations <- FALSE
if (!run_computations) {
  load("data/single_cell_obs.rda")
}

## functions
source("functions/ipp.R")
source("functions/distributional_anchor_regression.R")

# --- data and preprocessing ---------------------------------------------------

# data and preprocessing
data <- read.table("data/dataset_rpe1_99.csv", header = TRUE, sep = ",")
vnames <- c("non-targeting", colnames(data)[grepl("ENSG", colnames(data))])
data$hidden <- !(data$interventions %in% vnames)
rvar <- "ENSG00000173812"

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

# energy distance to select environments for prediction, compare to OLS
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

# compute energy distance and errors
nenv <- 50
if (run_computations) {
  edists <- ols_mse <- numeric(m)
  pb <- txtProgressBar(max = m)
  for (j in seq_len(m)) {
    intv_data <- data %>%
      filter(interventions == all_envs[j]) %>%
      select(-hidden, -interventions) %>%
      data.matrix()
    ols_mse[j] <- mean(
      (intv_data[, ncol(intv_data)] - intv_data[, -ncol(intv_data)] %*% clm)^2
    )
    edists[j] <- eqdist.e(rbind(obs_data, intv_data), sizes = c(nobs, nrow(intv_data)))
    setTxtProgressBar(pb, j)
  }
  close(pb)
  envs <- all_envs[order(edists, decreasing = TRUE)][seq_len(nenv)]
}

## format validation data
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
lambda <- c(0:5, seq(10, 70, 5))
nlambda <- length(lambda)

npar <- 2 * ncol(X_tr[[1]]) + 2
start_pop <- matrix(nrow = 200, ncol = npar, runif(npar * 200, -5, 5)) 
population <- function(object, ...) start_pop

if (run_computations) {
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
  save(list = ls(), file = "data/single_cell_obs.rda")
}

fit_logs$lambda <- fit_logs$lambda <- lambda
fit_logs$location <- fit_logs$location[, seq_along(lambda)]
fit_logs$scale <- fit_logs$scale[, seq_along(lambda)]
fit_logs$risk <- fit_logs$risk[seq_along(lambda)]
fit_logs$diff <- fit_logs$diff[seq_along(lambda)]
fit_logs$errors <- lapply(
  fit_logs$errors,
  function(x) x[, seq_along(lambda)]
)
fit_scrps$lambda <- fit_scrps$lambda <- lambda
fit_scrps$location <- fit_scrps$location[, seq_along(lambda)]
fit_scrps$scale <- fit_scrps$scale[, seq_along(lambda)]
fit_scrps$risk <- fit_scrps$risk[seq_along(lambda)]
fit_scrps$diff <- fit_scrps$diff[seq_along(lambda)]
fit_scrps$errors <- lapply(
  fit_scrps$errors,
  function(x) x[, seq_along(lambda)]
)

### get predicted location and scale parameter
pred_logs <- predict(fit_logs, cbind(1, X_vl))
pred_scrps <- predict(fit_scrps, cbind(1, X_vl))

### compute log score, squared error, sscrps
logs_logs <- logs_gaussian(Y_vl, pred_logs$location, pred_logs$scale)
logs_scrps <- scrps_gaussian(Y_vl, pred_logs$location, pred_logs$scale)
logs_sqerr <- (pred_logs$location - Y_vl)^2
scrps_logs <- logs_gaussian(Y_vl, pred_scrps$location, pred_scrps$scale)
scrps_scrps <- scrps_gaussian(Y_vl, pred_scrps$location, pred_scrps$scale)
scrps_sqerr <- (pred_scrps$location - Y_vl)^2

## distributional anchor regression
xi <- lambda
dar_fit <- dar_c_probit(
  X_tr[!(envs_interv %in% c(rvar, "non-targeting"))],
  Y_tr[!(envs_interv %in% c(rvar, "non-targeting"))],
  xi = xi,
  support = c(0.5, 5.5)
)

### compute log score, squared error, scrps
q <- seq(0.5, 5.5, 0.01)
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
      "Distributional anchor"
    ),
    labels = c(
      "IPP (LogS)",
      "IPP (SCRPS)",
      "Distributional anchor"
    ),
    ordered = TRUE
  ),
  each = 3
)
scores <- rep(factor(
  c("LogS", "SCRPS", "MSE"),
  levels =  c("LogS", "SCRPS", "MSE"),
  labels =  c("LogS", "SCRPS", "MSE"),
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
  mean_errors <- df %>%
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
      y = levels(scores)[min(i, 3)],
      color = element_blank(),
      fill = element_blank()
    ) +
    scale_y_continuous(
      labels = scales::number_format(accuracy = 0.01)
    ) +
    facet_grid(cols = vars(method), scales = "free_x")
  if (i >= 3) plt <- plt + coord_cartesian(ylim = c(0.1, 0.9))
  score_plots[[i]] <- plt
}

single_cell_test <- ggarrange(
  plotlist = score_plots,
  heights = c(1, 1, 1.05),
  ncol = 1
)

pdf("temporary_files/singleCellWithoutObservational.pdf", width = 8, height = 6)
print(single_cell_test)
dev.off()

#-------------------------------------------------------------------------------
# including response variable interventions for anchor related methods

## clean environment
rm(list = ls())
run_computations <- FALSE
if (!run_computations) {
  load("data/distr_anchor_rvar.rda")
}

## functions
source("functions/ipp.R")
source("functions/distributional_anchor_regression.R")

# --- data and preprocessing ---------------------------------------------------

# data and preprocessing
data <- read.table("data/dataset_rpe1_99.csv", header = TRUE, sep = ",")
vnames <- c("non-targeting", colnames(data)[grepl("ENSG", colnames(data))])
data$hidden <- !(data$interventions %in% vnames)
rvar <- "ENSG00000173812"

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

# energy distance to select environments for prediction, compare to OLS
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

# compute energy distance and errors
nenv <- 50
if (run_computations) {
  edists <- ols_mse <- numeric(m)
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
  envs <- all_envs[order(edists, decreasing = TRUE)][seq_len(nenv)]
}

## format validation data
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

## distributional anchor regression
xi <- lambda <- c(0:5, seq(10, 70, 5))
q <- seq(0.5, 5.5, 0.01)
q[1] <- q[1] + 1e-5
q[length(q)] <- q[length(q)] - 1e-5
if (run_computations) {
  dar_fit_without_rvar <- dar_c_probit(
    X_tr[!(envs_interv %in% c("non-targeting", rvar))],
    Y_tr[!(envs_interv %in% c("non-targeting", rvar))],
    xi = xi,
    support = c(0.5, 5.5)
  )
  dar_fit_with_rvar <- dar_c_probit(
    X_tr[!(envs_interv %in% c("non-targeting"))],
    Y_tr[!(envs_interv %in% c("non-targeting"))],
    xi = xi,
    support = c(0.5, 5.5)
  )
  dar_fits <- list(dar_fit_without_rvar, dar_fit_with_rvar)
  dar_scores <- vector("list", 2)
  for (i in seq_len(2)) {
    fit <- dar_fits[[i]]
    dar_prdint <- predint_dar(fit, X_vl, Y_vl)
    dar_scores[[i]] <- list(
      logs = logs_dar(fit, X_vl, Y_vl),
      scrps = scrps_dar(fit, X_vl, Y_vl, q),
      sqerr = sqerr_dar(fit, X_vl, Y_vl, q),
      crps = crps_dar(fit, X_vl, Y_vl, q),
      in_pi = dar_prdint[[1]],
      pi_width = dar_prdint[[2]]
    )
  }
  save(
    list = c("dar_scores", "envs"),
    file = "data/distr_anchor_rvar.rda"
  )
}

k <- 1
interv_rvar_fit <- c(FALSE, TRUE)
dar_results <- vector("list", 2 * length(dar_scores[[1]]))
for (j in seq_len(2)) {
  dar_scores_tmp <- dar_scores[[j]]
  for (i in seq_along(dar_scores_tmp)) {
    err <- lapply(
      asplit(dar_scores_tmp[[i]], 2),
      function(x) tapply(x, envs_long, mean)
    )
    dar_results[[k]] <- data.frame(
      env = names(unlist(err)),
      err = unname(unlist(err)),
      penalty = rep(lambda, each = length(err[[1]])),
      method = paste0("Distributional anchor"),
      score = names(dar_scores_tmp)[i],
      rvar_interv = interv_rvar_fit[j]
    )
    k <- k + 1
  }
}
dar_results <- do.call(rbind, dar_results)

## anchor regression and DRIG
an_dr_data <- 
  read.csv("data/anchor_drig.csv", header = TRUE, sep = ",")
an_dr_data <- an_dr_data %>%
  filter(gamma %in% lambda & interv_gene %in% envs) %>%
  select(-X) %>%
  rename(penalty = gamma, err = test_mse, env = interv_gene) %>%
  mutate(score = "sqerr", rvar_interv = FALSE)
an_dr_data_rvar_interv <- 
  read.csv("data/anchor_drig_rvar_interv.csv", header = TRUE, sep = ",")
an_dr_data_rvar_interv <- an_dr_data_rvar_interv %>%
  filter(gamma %in% lambda & interv_gene %in% envs) %>%
  select(-X) %>%
  rename(penalty = gamma, err = test_mse, env = interv_gene) %>%
  mutate(score = "sqerr", rvar_interv = TRUE)

## combine
df <- rbind(dar_results, an_dr_data, an_dr_data_rvar_interv) %>% 
  mutate(
    rvar_interv = ifelse(
      rvar_interv,
      "(with response intervened)",
      "(without)"
    ),
    method = paste0(method, "\n", rvar_interv),
    method = factor(
      method,
      labels = c(
        "Distributional anchor\n(with response intervened)",
        "Distributional anchor\n(without)",
        "DRIG\n(with response intervened)",
        "DRIG\n(without)",
        "Anchor regression\n(with response intervened)",
        "Anchor regression\n(without)"
      ),
      levels = c(
        "Distributional anchor\n(with response intervened)",
        "Distributional anchor\n(without)",
        "DRIG\n(with response intervened)",
        "DRIG\n(without)",
        "anchor regression\n(with response intervened)",
        "anchor regression\n(without)"   
      ),
      ordered = TRUE
    ),
    score = factor(
      score,
      levels = c("logs", "scrps", "crps", "sqerr", "in_pi", "pi_width"),
      labels = c(
        "LogS",
        "SCRPS",
        "CRPS",
        "MSE",
        "Coverage",
        "Length"
      ),
      ordered = TRUE
    )
  )


## quantiles of the scores
qs <- seq(0, 1, 0.1)
df_qs <- df %>%
  group_by(method, penalty, score) %>%
  summarise(
    out = list(tibble(qs, err = quantile(err, probs = qs, type = 1)))
  ) %>%
  unnest(cols = out)


## plots
df_qs$method_short <- substr(df_qs$method, 1, 4)
df$method_short <- substr(df$method, 1, 4)

scores_methods <- unique(df_qs[, c("method_short", "score")])
score_plots <- vector("list", length(scores_methods))

for (i in seq_len(nrow(scores_methods))) {
  df_tmp <- df_qs %>%
    filter(method_short == scores_methods$method_short[i] & 
      score == scores_methods$score[i])
  plt <- ggplot() +
    geom_line(
      data = filter(df_tmp, abs(qs - 0.5) < 1e-5),
      aes(x = penalty, y = err)
    )
  if (scores_methods$score[i] == "Coverage") {
    plt <- plt + 
      geom_hline(yintercept = 0.9, col = 2)
  }
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
  mean_errors <- df %>%
    filter(
      method_short == scores_methods$method_short[i] & 
        score == scores_methods$score[i]) %>%
    group_by(method, score, penalty) %>%
    summarise(err = mean(err))
  
  plt <- plt +
    geom_line(
      data = mean_errors,
      aes(x = penalty, y = err),
      lty = 5
    ) +
    labs(
      x = "Penalty",
      y = scores_methods$score[i],
      color = element_blank(),
      fill = element_blank()
    ) +
    scale_y_continuous(labels = scales::number_format(accuracy = 0.01)) +
    facet_wrap(.~method, ncol = 2) +
    theme_bw(base_size = 10)
  score_plots[[i]] <- plt
}


single_cell_test <- ggarrange(
  plotlist = score_plots,
  ncol = 2,
  nrow = 4
)

pdf("temporary_files/singleCellRvarAnchor.pdf", width = 8, height = 8)
print(single_cell_test)
dev.off()