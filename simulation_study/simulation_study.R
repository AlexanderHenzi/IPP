# --- global settings ----------------------------------------------------------

# packages
library(GA)
library(MASS)

# functions
source("ipp.R")

# parameters
id <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))
pars <- expand.grid(
  n_obs_env = c(50, 100, 150, 200, 250, 500, 1000),
  sim = seq_len(1000)
)
sim <- pars$sim[id]
n_x <- 5
n_obs_env <- pars$n_obs_env[id]
n_env <- n_x + 1
alpha_choose <- c(0.01, 0.05, 0.1)

# --- generate data ------------------------------------------------------------
Sigma_XY <- c(0.8, 0.4, 0.3, 0.2, 0.1) * (-1)^(seq_len(n_x) + 1)
cov_mat <- cbind(c(1, Sigma_XY), rbind(Sigma_XY, diag(n_x)))

set.seed(id)
beta <- runif(n_x, 0, 3)
gamma <- runif(n_x, 0, 0.5)
Y_e <- X_e <- Gamma_e <- vector("list", n_env)

for (i in seq_len(n_env - 1)) {
  eps_YX <- mvrnorm(n = n_obs_env, mu = rep(0, n_x + 1), Sigma = cov_mat)
  Gamma_e[[i]] <- diag(n_x) + 
    matrix(nrow = n_x, ncol = n_x, runif(n_x^2, -0.1, 0.1))
  X_e[[i]] <- t(tcrossprod(Gamma_e[[i]], eps_YX[, -1, drop = FALSE]))
  Y_e[[i]] <- c(X_e[[i]] %*% beta) + exp(c(X_e[[i]] %*% gamma)) * eps_YX[, 1]
}

alphas <- runif(n_env - 1, -1, 0)
alphas <- alphas / sum(abs(alphas))
eps_YX <- mvrnorm(n = n_obs_env, mu = rep(0, n_x + 1), Sigma = cov_mat)
Gamma_e[[n_env]] <- Reduce(
  `+`,
  mapply(
    function(alpha, mat) alpha * mat,
    alpha = alphas,
    mat = Gamma_e[seq_len(n_env - 1)],
    SIMPLIFY = FALSE
  ),
  init = matrix(nrow = n_x, ncol = n_x, 0)
)
X_e[[n_env]] <- t(tcrossprod(Gamma_e[[n_env]], eps_YX[, -1, drop = FALSE]))
Y_e[[n_env]] <- c(X_e[[n_env]] %*% beta) + exp(c(X_e[[n_env]] %*% gamma)) *
  eps_YX[, 1]

# --- fit ----------------------------------------------------------------------

## penalization parameter
lambda <- seq(0, 30, 0.5)

## for optimization
n_par <- 2 * ncol(X_e[[1]])
n_pop <- 100
start_pop <- matrix(nrow = n_pop, ncol = n_par, runif(n_par * n_pop, -5, 5)) 
population <- function(object, ...) start_pop

score <- function(y, l, s) -dnorm(y, mean = l, sd = s, log = TRUE)
# score <- function(y, l, s) {
#   z <- (l - y) / s
#   out <- sqrt(pi) * dnorm(z) + sqrt(pi) * z * 
#     (2 * pnorm(z) - 1) / 2 + log(2 * s / sqrt(pi)) / 2
#   out
# }

## fit
fit <- ipp(
  X = X_e,
  Y = Y_e,
  score = score,
  lambda = lambda,
  type = "real-valued",
  lower = rep(-5, n_par),
  upper = rep(5, n_par),
  optim = TRUE,
  optimArgs = list(method = "CG", poptim = 0.1, maxit = 200),
  monitor = FALSE,
  popSize = n_pop,
  maxiter = 400,
  population = population,
  seed = id + 20
)

## find out which lambdas would be chosen
n_lambda <- length(lambda)
chosen_lambda <- numeric(length(alpha_choose))
pvals <- numeric(n_lambda)
for (j in seq_len(n_lambda)) {
  pvals[j] <- oneway.test(
    err ~ env,
    data = do.call(
      rbind,
      mapply(
        function(err, env) data.frame(err = err, env = env),
        err = lapply(fit$errors, function(x) x[, j]),
        env = seq_len(n_env),
        SIMPLIFY = FALSE
      )
    )
  )$p.value
}
for (j in seq_along(alpha_choose)) {
  if (any(pvals > alpha_choose[j])) {
    chosen_lambda[j] <- lambda[min(which(pvals > alpha_choose[j]))]
  } else {
    chosen_lambda[j] <- lambda[length(lambda)]
  }
}

# --- error on test environments -----------------------------------------------
n_test <- 60000

## same as training distribution
for (i in seq_len(n_env)) {
  eps_YX <- mvrnorm(n = n_obs_env, mu = rep(0, n_x + 1), Sigma = cov_mat)
  Gamma_e[[i]] <- diag(n_x) + 
    matrix(nrow = n_x, ncol = n_x, runif(n_x^2, -0.1, 0.1))
  X_e[[i]] <- t(tcrossprod(Gamma_e[[i]], eps_YX[, -1, drop = FALSE]))
  Y_e[[i]] <- c(X_e[[i]] %*% beta) + exp(c(X_e[[i]] %*% gamma)) * eps_YX[, 1]
}

alphas <- runif(n_env - 1, -1, 0)
alphas <- alphas / sum(abs(alphas))
eps_YX <- mvrnorm(n = n_obs_env, mu = rep(0, n_x + 1), Sigma = cov_mat)
Gamma_e[[n_env]] <- Reduce(
  `+`,
  mapply(
    function(alpha, mat) alpha * mat,
    alpha = alphas,
    mat = Gamma_e[seq_len(n_env - 1)],
    SIMPLIFY = FALSE
  ),
  init = matrix(nrow = n_x, ncol = n_x, 0)
)
X_e[[n_env]] <- t(tcrossprod(Gamma_e[[n_env]], eps_YX[, -1, drop = FALSE]))
Y_e[[n_env]] <- c(X_e[[n_env]] %*% beta) + exp(c(X_e[[n_env]] %*% gamma)) *
  eps_YX[, 1]

X <- do.call(rbind, X_e)
Y <- unlist(Y_e)
preds <- predict(fit, X)
errs <- score(Y, preds$location, preds$scale)

prediction_error <- colMeans(errs)
prediction_error

## mean shift
shift <- runif(n_x, -5, 5)
shift <- (shift - sum(shift * gamma) / sum(gamma^2) * gamma)
eps_YX <- mvrnorm(n = n_test, mu = rep(0, n_x + 1), Sigma = cov_mat)
X <- t(t(eps_YX[, -1]) + shift)
Y <-  c(X %*% beta) + exp(c(X %*% gamma)) * eps_YX[, 1]
preds <- predict(fit, X)
errs <- score(Y, preds$location, preds$scale)

mean_shift_error <- colMeans(errs)
mean_shift_error

## high variance
eps_YX <- mvrnorm(n = n_test, mu = rep(0, n_x + 1), Sigma = cov_mat)
X <- 1.5 * eps_YX[, -1]
Y <-  c(X %*% beta) + exp(c(X %*% gamma)) * eps_YX[, 1]
preds <- predict(fit, X)
errs <- score(Y, preds$location, preds$scale)

high_variance_error <- colMeans(errs)
high_variance_error

## lower variance
eps_YX <- mvrnorm(n = n_test, mu = rep(0, n_x + 1), Sigma = cov_mat)
X <- eps_YX[, -1] / 3
Y <-  c(X %*% beta) + exp(c(X %*% gamma)) * eps_YX[, 1]
preds <- predict(fit, X)
errs <- score(Y, preds$location, preds$scale)

low_variance_error <- colMeans(errs)
low_variance_error

## stronger correlation and high variance
eps_YX <- mvrnorm(n = n_test, mu = rep(0, n_x + 1), Sigma = cov_mat)
Gamma <- diag(n_x) +
  matrix(nrow = n_x, ncol = n_x, runif(n_x^2, -0.75, 0.75))
X <-  t(tcrossprod(Gamma, eps_YX[, -1]))
Y <-  c(X %*% beta) + exp(c(X %*% gamma)) * eps_YX[, 1]
preds <- predict(fit, X)
errs <- score(Y, preds$location, preds$scale)

corr_error <- colMeans(errs)
corr_error

# --- export results -----------------------------------------------------------
mean_err <- fit$loss
var_err <- fit$diff
location <- fit$location
scale <- fit$scale

save(
  list = c(
    "sim",
    "n_obs_env",
    "mean_err",
    "var_err",
    "location",
    "scale",
    "beta",
    "gamma",
    "prediction_error",
    "mean_shift_error",
    "high_variance_error",
    "low_variance_error",
    "corr_error",
    "lambda",
    "chosen_lambda",
    "alpha_choose"
  ),
  file = paste0("simulations_", id, ".rda")
)
