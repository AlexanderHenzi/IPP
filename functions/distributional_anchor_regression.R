# parts of this code are copied and adapted from
#   https://github.com/LucasKook/distributional-anchor-regression

#' Fit distributional anchor regression for a single penalty parameter
#' 
#' @param m0 base conditional transformation model.
#' @param xi penalty parameter.
#' @param data list containing outcome variable, covariates, and matrix for 
#'     anchor variables.
#' 
#' @details 
#' The function is only used internally in dar_c_probit.
#' 
#' @return 
#' The estimated coefficients.
BoxCox_anchor <- function(m0, xi, data) {
  A <- data$A
  dat <- data.frame(y = data$Y, x = data$X)
  trdat <- anchor:::.get_tram_data(m0)
  xe <- data$X[trdat$exact$which, , drop = FALSE]
  nth <- trdat$npar
  nb <- ncol(data$X)
  theta <- Variable(nth)
  beta <- Variable(nb)
  Pia <- A %*% solve(t(A) %*% A) %*% t(A)
  resids <- (trdat$exact$ay %*% theta - xe %*% beta)
  v <- Pia %*% resids
  z <- trdat$exact$ay %*% theta - xe %*% beta
  const <- -log(sqrt(2 * pi)) * trdat$nobs
  ll <- const - sum(z^2/2) + sum_entries(log(trdat$exact$aypr %*% theta))
  obj <- -ll + xi * sum_entries(power(v, 2))
  const <- list(trdat$const$ui %*% theta >= trdat$const$ci)
  prob <- Problem(Minimize(obj), constraints = const)
  res <- solve(prob)
  class(res) <- "BC"
  return(res)
}

#' Transform data to list form with model.matrix
#' 
#' @param fml formula for generating the model matrix.
#' @param data data frame containing covariates and response variable.
#' 
#' @details 
#' The function is only used internally in dar_c_probit.
#' 
#' @return 
#' The data in the required format.
to_list <- function(fml, data) {
  mf <- model.frame(fml, data)
  Y <- model.response(mf)
  X <- anchor:::.rm_int(model.matrix(fml, data))
  list(Y = matrix(Y, ncol = 1), X = X)
}

#' Fit distributional anchor regression with probit link
#' 
#' @param X list of covariate matrices. Categorical variabeles must be encoded,
#'     and an intercept has to be included if desired.
#' @param Y list of response variable vectors.
#' @param xi vector of penalization parameter values.
#' @param support vector of length two, containing the support endpoints for
#'     the response variable. Make this large enough so that it covers all
#'     values of the response variable in training and test data. Passed to
#'     \code{\link[tram]{BoxCox}.
#'     
#' @return 
#' A list containing the fitted models for all values of \code{xi}, the training
#' data, and the given penalization parameters and support.
dar_c_probit <- function(
    X,
    Y,
    xi,
    support) {
  data <- data.frame(do.call(rbind, X))
  colnames(data) <- colnames(X_vl) <- paste0("X", seq_len(ncol(data)))
  newdata <- list(Y = Y_vl, X = X_vl)
  data <- cbind(Y = unlist(Y), data)
  fml <- as.formula(paste0(
    "Y ~ ", paste0("X", seq_len(ncol(data) - 1), collapse = " + ")
  ))
  data <- cbind(
    data,
    A = factor(rep(seq_along(X), times = sapply(X, nrow)))
  )
  anchor <- "A"
  A <- model.matrix(as.formula(paste("~ ", anchor)), data = data)
  X <- model.matrix(fml, data = data)[, -1L]
  m0 <- BoxCox(update(fml, . ~ 1), data = data, support = support)
  m02 <- as.mlt(BoxCox(fml, data = data, support = support))
  ltrain <- to_list(fml, data = data)
  ltrain$A <- A
  m <- length(xi)
  models <- vector("list", m)
  pb <- txtProgressBar(max = m)
  for (j in seq_len(m)) {
    ma <- BoxCox_anchor(m0 = m0, xi = xi[j], data = ltrain)
    coefs <- unlist(ma[1:2])
    names(coefs) <- names(coef(m02))
    m_tmp <- m02
    coef(m_tmp) <- coefs
    models[[j]] <- m_tmp
    setTxtProgressBar(pb, value = j)
  }
  close(pb)
  out <- list(models = models, xi = xi, X = X, Y = Y, support = support)
  structure(out, class = "dar_fit")
}

#' Logarithmic score for distributional anchor regression
#' 
#' @param fit output of \code{dar_c_probit}.
#' @param X_vl covariates for validation data.
#' @param Y_vl response variable for validation data.
#' 
#' @return
#' A matrix of size \code{nrow(X_vl)} times the number of penalization 
#' parameters in \code{fit}, containing the logarithmic scores.
logs_dar <- function(fit, X_vl, Y_vl) {
  m <- length(fit$models)
  out <- matrix(nrow = length(Y_vl), ncol = m)
  d <- ncol(X_vl)
  ncoef <- length(coef(fit$models[[1]]))
  ydat <- data.frame(Y = Y_vl)
  X_vl <- data.matrix(X_vl)
  for (j in seq_len(m)) {
    coef_tmp <- coef(fit$models[[j]])
    a_tmp <- coef_tmp[seq_len(ncoef - d)]
    b_tmp <- coef_tmp[(ncoef - d + 1):ncoef]
    yb <- model.matrix(fit$models[[j]]$model$bases$response, data = ydat)
    deriv <- 1L
    names(deriv) <- "Y"
    ybp <- model.matrix(fit$models[[j]]$model$bases$response, data = ydat,
                        deriv = deriv)
    out[, j] <- -dnorm(c(yb %*% a_tmp) -
                   c(X_vl %*% b_tmp), log = TRUE) - log(c(ybp %*% a_tmp))
  }
  out
}

#' Squared error for distributional anchor regression
#' 
#' @param fit output of \code{dar_c_probit}.
#' @param X_vl covariates for validation data.
#' @param Y_vl response variable for validation data.
#' @param q grid over the support of the outcome variable. Should be contained
#'     in the support of \code{fit}. Is used to approximate the conditional
#'     CDFs and the mean.
#' 
#' @return
#' A matrix of size \code{nrow(X_vl)} times the number of penalization 
#' parameters in \code{fit}, containing the squared error for the mean
#' predictions.
sqerr_dar <- function(fit, X_vl, Y_vl, q) {
  m <- length(fit$models)
  out <- matrix(nrow = length(Y_vl), ncol = length(fit$models))
  X_vl <- as.data.frame(X_vl)
  colnames(X_vl) <- paste0("X", seq_len(ncol(fit$models[[1]]$data[-1])))
  nq <- length(q)
  for (j in seq_len(m)) {
    ps <- predict(
      fit$models[[j]],
      type = "distribution",
      q = q,
      newdata = X_vl
    )
    ps <- t(ps) / ps[nq, ]
    if (!is.matrix(ps) | nrow(ps) != length(Y_vl) | ncol(ps) != nq) {
      stop("predict tram error")
    } else {
      out[, j] <- (Y_vl - c((ps - cbind(0, ps[, -nq])) %*% q))^2
    }
  }
  out
}

#' Prediction intervals for distributional anchor regression
#' 
#' @param fit output of \code{dar_c_probit}.
#' @param X_vl covariates for validation data.
#' @param Y_vl response variable for validation data.
#' @param conf.level desired coverage
#' 
#' @return
#' A matrix of size \code{nrow(X_vl)} times the number of penalization 
#' parameters in \code{fit}, containing the squared error for the mean
#' predictions.
predint_dar <- function(fit, X_vl, Y_vl, conf.level = 0.9) {
  m <- length(fit$models)
  out <- vector("list", 2)
  out[[1]] <- out[[2]] <- matrix(nrow = length(Y_vl), ncol = length(fit$models))
  X_vl <- as.data.frame(X_vl)
  colnames(X_vl) <- paste0("X", seq_len(ncol(fit$models[[1]]$data[-1])))
  alpha <- (1 - conf.level) / 2
  pb <- txtProgressBar(max = m)
  for (j in seq_len(m)) {
    setTxtProgressBar(pb, j)
    lwr <- predict(
      fit$models[[j]],
      type = "quantile",
      prob = alpha,
      newdata = X_vl
    )
    if (is.data.frame(lwr)) lwr <- lwr$approxy
    upr <- predict(
      fit$models[[j]],
      type = "quantile",
      prob = 1 - alpha,
      newdata = X_vl
    )
    if (is.data.frame(upr)) upr <- upr$approxy
    out[[1]][, j] <- (Y_vl > lwr & Y_vl < upr)
    out[[2]][, j] <- upr - lwr
  }
  close(pb)
  out
}

#' Scrps for distributional anchor regression
#' 
#' @param fit output of \code{dar_c_probit}.
#' @param X_vl covariates for validation data.
#' @param Y_vl response variable for validation data.
#' @param q grid over the support of the outcome variable. Should be contained
#'     in the support of \code{fit}. Is used to approximate the conditional
#'     CDFs for the computation of the SCRPS.
#' 
#' @return
#' A matrix of size \code{nrow(X_vl)} times the number of penalization 
#' parameters in \code{fit}, containing the SCRPS.
scrps_dar <- function(fit, X_vl, Y_vl, q) {
  m <- length(fit$models)
  out <- matrix(nrow = length(Y_vl), ncol = length(fit$models))
  X_vl <- as.data.frame(X_vl)
  colnames(X_vl) <- paste0("X", seq_len(ncol(fit$models[[1]]$data[-1])))
  nq <- length(q)
  dq <- diff(q)
  for (j in seq_len(m)) {
    ps <- predict(
      fit$models[[j]],
      type = "distribution",
      q = q,
      newdata = X_vl
    )
    ps <- t(ps) / ps[nq, ]
    if (!is.matrix(ps) | nrow(ps) != length(Y_vl) | ncol(ps) != nq) {
      stop("predict tram error")
    }
    mean_diff <- 
      c((ps[, -1] * (1 - ps[, -1]) + ps[, -nq] * (1 - ps[, -nq])) %*% dq)
    mae <- rowSums(abs(outer(Y_vl, q, FUN = "-")) * (ps - cbind(0, ps[, -nq])))
    out[, j] <- mae / mean_diff + log(mean_diff) / 2
  }
  out
}

#' crps for distributional anchor regression
#' 
#' @param fit output of \code{dar_c_probit}.
#' @param X_vl covariates for validation data.
#' @param Y_vl response variable for validation data.
#' @param q grid over the support of the outcome variable. Should be contained
#'     in the support of \code{fit}. Is used to approximate the conditional
#'     CDFs for the computation of the SCRPS.
#' 
#' @return
#' A matrix of size \code{nrow(X_vl)} times the number of penalization 
#' parameters in \code{fit}, containing the SCRPS.
crps_dar <- function(fit, X_vl, Y_vl, q) {
  m <- length(fit$models)
  out <- matrix(nrow = length(Y_vl), ncol = length(fit$models))
  X_vl <- as.data.frame(X_vl)
  colnames(X_vl) <- paste0("X", seq_len(ncol(fit$models[[1]]$data[-1])))
  nq <- length(q)
  dq <- diff(q)
  for (j in seq_len(m)) {
    ps <- predict(
      fit$models[[j]],
      type = "distribution",
      q = q,
      newdata = X_vl
    )
    ps <- t(ps) / ps[nq, ]
    if (!is.matrix(ps) | nrow(ps) != length(Y_vl) | ncol(ps) != nq) {
      stop("predict tram error")
    }
    mean_diff <- 
      c((ps[, -1] * (1 - ps[, -1]) + ps[, -nq] * (1 - ps[, -nq])) %*% dq)
    mae <- rowSums(abs(outer(Y_vl, q, FUN = "-")) * (ps - cbind(0, ps[, -nq])))
    out[, j] <- mae - mean_diff / 2
  }
  out
}