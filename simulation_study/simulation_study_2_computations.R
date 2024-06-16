# --- global settings ----------------------------------------------------------

# packages
library(GA)
library(reticulate)
library(scoringRules)

# functions
source("functions/ipp.R")
reticulate::source_python("functions/anchor_drig.py")
reticulate::source_python("functions/cputils.py")

# parameters
id <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))
response_gene <- "ENSG00000173812"
gamma_drig <- seq(0.1, 30, 0.1)
lambda_ipp <- c(seq(0, 10, 0.5), 11:40)
n_pop_ipp <- 400
maxit_ipp <- 500
maxit_optim_ipp <- 200
poptim_ipp <- 0.1
alpha <- 0.1
rho_cp <- c(0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04)

# seed
set.seed(id)

# time
start_time <- Sys.time()

return np.sum(w*((Ss<=t)-m(x,t)))/np.sum(w)+np.sum(m(shiftx,t))/shiftx.shape[0]


#-------------------------------------------------------------------------------
# scores

logs_norm <- function(y, l, s) -dnorm(y, mean = l, sd = s, log = TRUE)
crps_norm <- function(y, l, s) scoringRules::crps_norm(y = y, mean = l, sd = s)
scrps_norm <- function(y, l, s) {
  z <- (l - y) / s
  out <- sqrt(pi) * dnorm(z) + sqrt(pi) * z * 
    (2 * pnorm(z) - 1) / 2 + log(2 * s / sqrt(pi)) / 2
  out
}
sqerr_norm <- function(y, l, s) (y - l)^2
in_pi <- function(y, l, s) 
  (y <= qnorm(1 - alpha  / 2, l, s) & y >= qnorm(alpha / 2, l, s))
pi_width <- function(y, l, s) 
  qnorm(1 - alpha / 2, l, s) - qnorm(alpha / 2, l, s)

#-------------------------------------------------------------------------------
# load and prepare data

## read and split data
test_data <- read.csv(
  "data/simulation_2_test_data.txt",
  sep = ";",
  header = TRUE
)
training_data <- read.csv(
  "data/simulation_2_training_data.txt",
  sep = ";",
  header = TRUE
)
test_data <- test_data[test_data$sim == id, -ncol(test_data)]
training_data <- training_data[training_data$sim == id, -ncol(training_data)]
test_data <- split(
  test_data[, -ncol(test_data)],
  test_data$intervention
)
training_data <- split(
  training_data[, -ncol(training_data)],
  training_data$intervention
)

## data preparation for anchor/drig: observational first; center
test_data <- rev(test_data)
training_data <- rev(training_data)
envs <- names(test_data)
envs_long <- rep(envs, times = sapply(test_data, nrow))
test_data <- unname(test_data)
training_data <- unname(training_data)

observational_means <- colMeans(training_data[[1]])
training_data <- lapply(
  training_data,
  function(x) t(t(unname(data.matrix(x))) - observational_means)
)
test_data <- lapply(
  test_data,
  function(x) t(t(unname(data.matrix(x))) - observational_means)
)

#-------------------------------------------------------------------------------
# anchor regression and drig

n_gamma <- length(gamma_drig)
n_env <- length(envs)

mse_anchor <- mse_drig <- matrix(nrow = n_env, ncol = n_gamma, 0)

for (j in seq_len(n_gamma)) {
  mse_drig[, j] <- mse_drig[, j] +
    unlist(test_mse_list(
      test_data,
      est(training_data, method="drig", gamma=gamma_drig[j]))
    )
  mse_anchor[, j] <- mse_anchor[, j] +
    unlist(test_mse_list(
      test_data,
      est(training_data, method="anchor", gamma=gamma_drig[j]))
    )
}

results_anchor <- data.frame(
  score = "mse",
  value = c(mse_anchor),
  env = rep(envs, n_gamma),
  penalty = rep(gamma_drig, each = n_env),
  method = "anchor",
  sim = id
)
results_drig <- data.frame(
  score = "mse",
  value = c(mse_drig),
  env = rep(envs, n_gamma),
  penalty = rep(gamma_drig, each = n_env),
  method = "drig",
  sim = id
)

#-------------------------------------------------------------------------------
# ipp

## prepare data
X_training_ipp <- lapply(training_data, function(x) cbind(1, x[, -ncol(x)]))
Y_training_ipp <- lapply(training_data, function(x) x[, ncol(x)])
X_test_ipp <- cbind(1, do.call(rbind, test_data)[, -ncol(X_training_ipp[[1]])])
Y_test_ipp <- do.call(rbind, test_data)[, ncol(X_test_ipp)]

nlambda <- length(lambda_ipp)
npar <- 2 * ncol(X_training_ipp[[1]])
start_pop <- matrix(
  nrow = n_pop_ipp,
  ncol = npar,
  runif(npar * n_pop_ipp, -5, 5)
) 
population <- function(object, ...) start_pop

fit_logs <- ipp(
  X = X_training_ipp,
  Y = Y_training_ipp,
  lambda = lambda_ipp,
  score = logs_norm,
  type = "real-valued",
  nBits = NULL,
  lower = rep(-5, npar),
  upper = rep(5, npar),
  optim = TRUE,
  optimArgs = list(method = "CG", poptim = poptim_ipp, maxit = maxit_optim_ipp),
  monitor = FALSE,
  popSize = n_pop_ipp,
  maxiter = maxit_ipp,
  population = population,
  seed = 20230531
)
fit_crps <- ipp(
  X = X_training_ipp,
  Y = Y_training_ipp,
  lambda = lambda_ipp,
  score = crps_norm,
  type = "real-valued",
  nBits = NULL,
  lower = rep(-5, npar),
  upper = rep(5, npar),
  optim = TRUE,
  optimArgs = list(method = "CG", poptim = poptim_ipp, maxit = maxit_optim_ipp),
  monitor = FALSE,
  popSize = n_pop_ipp,
  maxiter = maxit_ipp,
  population = population,
  seed = 20230531
)
fit_scrps <- ipp(
  X = X_training_ipp,
  Y = Y_training_ipp,
  lambda = lambda_ipp,
  score = scrps_norm,
  type = "real-valued",
  nBits = NULL,
  lower = rep(-5, npar),
  upper = rep(5, npar),
  optim = TRUE,
  optimArgs = list(method = "CG", poptim = poptim_ipp, maxit = maxit_optim_ipp),
  monitor = FALSE,
  popSize = n_pop_ipp,
  maxiter = maxit_ipp,
  population = population,
  seed = 20230531
)

ipp_fits <- list(logs = fit_logs, scrps = fit_scrps, crps = fit_crps)
ipp_preds <- lapply(
  ipp_fits,
  function(x) predict(x, X_test_ipp)
)
scores_list <- list(
  crps = crps_norm,
  logs = logs_norm,
  scrps = scrps_norm,
  mse = sqerr_norm,
  in_pi = in_pi,
  pi_width = pi_width
)
ipp_scores <- vector("list", length(scores_list) * length(ipp_preds))
k <- 1
for (i in seq_along(ipp_preds)) {
  pred <- ipp_preds[[i]]
  for (j in seq_along(scores_list)) {
    score_fun <- scores_list[[j]]
    score <- score_fun(Y_test_ipp, pred$location, pred$scale)
    score <- tapply(
      c(score),
      list(
        env = rep(envs_long, nlambda),
        penalty = rep(lambda_ipp, each = length(envs_long))
      ),
      mean
    )
    score <- score[envs, ]
    ipp_scores[[k]] <- data.frame(
      score = names(scores_list)[j],
      value = c(score),
      env = rep(envs, nlambda),
      penalty = rep(lambda_ipp, each = n_env),
      method = paste0("ipp_", names(ipp_preds)[i]),
      sim = id
    )
    k <- k + 1
  }
}
results_ipp <- do.call(rbind, ipp_scores)

#-------------------------------------------------------------------------------
# conformal prediction

training <- training_data[[1]]
n_rho <- length(rho_cp)

results_cp <- vector("list", n_rho * n_env)
k <- 1
for (i in seq_len(n_rho)) {
  for (j in seq_len(n_env)) {
    results_tmp <- fit_pred_cp(
      training,
      test_data[[j]],
      rho_cp[i],
      alpha,
      as.integer(id)
    )
    results_tmp["env"] <- envs[j]
    results_cp[[k]] <- results_tmp
    k <- k + 1
  }
}
results_cp <- do.call(rbind, results_cp)

#-------------------------------------------------------------------------------
# other methods?

#-------------------------------------------------------------------------------
# export results

results <- rbind(results_anchor, results_drig, results_ipp, results_cp)
write.table(
  x = results,
  file = "results.txt",
  sep = ";"
)

Sys.time() - start_time