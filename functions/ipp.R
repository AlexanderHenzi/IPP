#' Errors of distributional regression prediction
#' 
#' @param X list of covariate matrices. Categorical variabeles must be encoded,
#'     and an intercept has to be included if desired.
#' @param Y list of response variable vectors.
#' @param score scoring rule, taking arguments \code{y} for the outcome
#'     variable, \code{l} for the location parameter, and \code{s} for the
#'     scale parameter, in this order.
#' @param location matrix of location parameters, with rows for the parameters
#'     and columns for penalization values.
#' @param scale matrix of scale parameters, with rows for the parameters and
#'     columns for penalization values.
#'     
#' @return 
#' A list containing matrices with the errors, vectors of the mean errors, and
#' of the mean squared differences between errors in different environemtns.
loss_path <- function(X, Y, score, location, scale) {
  nenv <- length(X)
  m <- ncol(location)
  loss <- diff <- numeric(m)
  errors <- matrix(nrow = sum(lengths(Y)), ncol = m)
  for (j in seq_len(m)) {
    lctn <- lapply(X, function(x) c(x %*% location[, j]))
    scl <- lapply(X, function(x) c(exp(x %*% scale[, j])))
    scores <- mapply(
      function(y, l, s) {score(y, l, s)},
      y = Y,
      l = lctn,
      s = scl,
      SIMPLIFY = FALSE
    )
    errors[, j] <- unlist(scores)
    scores <- sapply(scores, mean)
    loss[j] <- mean(scores)
    diff[j] <- var(scores) * (nenv - 1) / nenv
  }
  errors <- lapply(
    split(as.data.frame(errors), rep(seq_along(Y), times = lengths(Y))),
    function(df) unname(data.matrix(df))
  )
  list(risk = loss, diff = diff, errors = errors)
}

#' Target function for optimization in IPP
#' 
#' @param X list of covariate matrices. Categorical variabeles must be encoded,
#'     and an intercept has to be included if desired.
#' @param Y list of response variable vectors.
#' @param score log density function, taking arguments \code{y} for the
#'     outcome variable, \code{loc} for the location parameter, and \code{scl}
#'     for the scale parameter.
#' @param lambda penalization parameter.
#' @param par vector of parameters at which the target function is evaluated.
#' 
#' @return 
#' Value of target function at the given parameters.
ipp_target <- function(X, Y, score, lambda, par) {
  d <- ncol(X[[1]])
  lpar <- par[seq_len(d)]
  spar <- par[-seq_len(d)]
  location <- lapply(X, function(x) c(x %*% lpar))
  scale <- lapply(X, function(x) c(exp(x %*% spar)))
  scores <- mapply(
    function(y, l, s) score(y, l, s),
    y = Y,
    l = location,
    s = scale,
    SIMPLIFY = FALSE
  )
  nenvs <- length(X)
  scores <- sapply(scores, mean)
  mean(scores) + lambda * var(scores) * (nenvs - 1) / nenvs
}

#' Fit IPP to training data
#' 
#' @param X list of covariate matrices. Categorical variabeles must be encoded,
#'     and an intercept has to be included if desired.
#' @param Y list of response variable vectors.
#' @param score scoring rule, taking arguments \code{y} for the outcome
#'     variable, \code{l} for the location parameter, and \code{s} for the
#'     scale parameter, in this order.
#' @param lambda vector of penalization parameter values.
#' @param ... further arguments passed to the optimizer \code{\link[GA]{ga}.
#' 
#' @return
#' A list containing X, Y, the estimated location parameters for all values
#' of lambda, and the corresponding errors, mean errors across environments,
#' and squared differences of errors between environments.
ipp <- function(X, Y, score, lambda, ...) {
  d <- ncol(X[[1]])
  m <- length(lambda)
  location <- scale <- matrix(nrow = d, ncol = m)
  opt <- ga(
    fitness = function(par) {
      -ipp_target(X = X, Y = Y, score = score, lambda = lambda[1], par = par)
    },
    ...
  )
  opt <- opt@solution[1, ]
  location[, 1] <- opt[seq_len(d)]
  scale[, 1] <- opt[-seq_len(d)]
  pb <- txtProgressBar(max = m, initial = 1)
  if (m > 1) {
    for (j in 2:m) {
      opt <- ga(
        fitness = function(par) {
          -ipp_target(X = X, Y = Y, score = score, lambda = lambda[j], par = par)
        },
        ...
      )
      opt <- opt@solution[1, ]
      location[, j] <- opt[seq_len(d)]
      scale[, j] <- opt[-seq_len(d)]
      setTxtProgressBar(pb, j)
    }
  }
  close(pb)
  out <- list(
    location = location,
    scale = scale,
    lambda = lambda,
    X = X,
    Y = Y
  )
  path <- loss_path(X, Y, score, location, scale)
  structure(c(out, path), class = "ipp_fit")
}

#' Predict from ipp_fit
#' 
#' @param fit output of ipp.
#' @param new_X new covariate matrix.
#' 
#' @return
#' A list containing matrices of location and scale parameters.
predict.ipp_fit <- function(fit, new_X) {
  loc <- new_X %*% fit$location
  sc <- exp(new_X %*% fit$scale)
  list(location = loc, scale = sc)
}