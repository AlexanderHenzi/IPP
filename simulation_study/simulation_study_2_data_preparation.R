# --- global settings ----------------------------------------------------------

# packages
library(ggplot2)

# parameters
n_obs_intv <- 80
n_obs_observ <-  1000
n_sim <- 500

# settings
theme_set(theme_bw(base_size = 12))

# data generation parameters
response_gene <- "ENSG00000173812"
hidden_confounders <- c(
  "ENSG00000187514",
  "ENSG00000075624",
  "ENSG00000147604",
  "ENSG00000110700" 
)

# seed
set.seed(20230531)

#-------------------------------------------------------------------------------
# functions

generate_data <- function(
    data,
    n_obs_intv,
    n_obs_observ,
    residuals,
    beta,
    gamma) {
  dfs <- 
    lapply(data, function(x) x[sample(nrow(x), n_obs_intv, replace = TRUE), ])
  dfs[[which(names(dfs) == "non-targeting")]] <- 
    data[[which(names(data) == "non-targeting")]][sample(
      nrow(data[[which(names(data) == "non-targeting")]]),
      n_obs_observ,
      replace = TRUE
    ), ]
  n <- sum(sapply(dfs, nrow))
  residuals <- split(
    sample(residuals, n, replace = TRUE),
    rep(names(dfs), times = sapply(dfs, nrow))
  )
  locations <- lapply(dfs, function(x) beta[1] + x %*% beta[-1])
  scales <- lapply(dfs, function(x) exp(gamma[1] + x %*% gamma[-1]))
  Y <- mapply(
    function(l, s, r) l + s * r,
    l = locations,
    s = scales,
    r = residuals,
    SIMPLIFY = FALSE
  )
  X <- lapply(dfs, function(x) x[, !colnames(x) %in% hidden_confounders])
  XY <- mapply(
    function(x, y, intrv) {
      out <- as.data.frame(x)
      out[response_gene] <- c(y)
      out["intervention"] <- intrv
      out
    },
    x = X,
    y = Y,
    intrv = names(dfs),
    SIMPLIFY = FALSE
  )
  do.call(rbind, XY)
}

#-------------------------------------------------------------------------------
# read and prepare data

## read data
all_data <- read.table("data/dataset_rpe1_99.csv", header = TRUE, sep = ",")

## select relevant environments
all_data <- all_data[all_data$interventions %in% colnames(all_data) |
 all_data$interventions == "non-targeting", ]

## covariate data for environments
covariate_data <- split(
  all_data[, !colnames(all_data) %in% c("interventions", response_gene)],
  all_data[, "interventions"]
)
covariate_data <- lapply(covariate_data, data.matrix)
covariate_data <- covariate_data[names(covariate_data) != response_gene]

## fit parameters on observational data
fit_data <- all_data[all_data$interventions == "non-targeting", -ncol(all_data)]

fml <- as.formula(paste0(response_gene, " ~ ."))
location_fit <- lm(fml, data = fit_data)
res <- residuals(location_fit)

scale_fit_data <- fit_data
scale_fit_data[response_gene] <- log(abs(res))
scale_fit <- lm(fml, data = scale_fit_data)

res <- (fit_data[, response_gene] - fitted(location_fit)) / 
  exp(fitted(scale_fit))
res <- (res - mean(res)) / sd(res)

beta <- coef(location_fit)
gamma <- coef(scale_fit)
beta_observed <- beta[!names(beta) %in% hidden_confounders]
gamma_observed <- gamma[!names(gamma) %in% hidden_confounders]

resqq <- as.data.frame(qqnorm(res))
qqline <- coef(lm(y ~ x, data = resqq))

resqqplt <- ggplot() +
  geom_point(
    data = resqq,
    aes(x = x, y = y)
  ) +
  geom_abline(intercept = qqline[1], slope = qqline[2], col = "red") +
  labs(x = "Theoretical quantiles", y = "Sample quantiles")

pdf("simulation_2_qq.pdf", width = 8, height = 8)
print(resqqplt)
dev.off()

## table in article
info_table <- data.frame(
  gene = rev(names(beta)),
  treatment = ifelse(
    rev(names(beta)) %in% hidden_confounders,
    "hidden",
    "observed"
  ),
  location = rev(beta),
  scale = rev(gamma)
)
info_table <- rbind(
  info_table,
  data.frame(
    gene = "non-targeting",
    treatment = NA,
    location = NA,
    scale = NA
  )
)
ns <- sapply(covariate_data, nrow)
info_table$n <- ns[info_table$gene]
info_table$location <- sprintf("%.3f", info_table$location)
info_table$scale <- sprintf("%.3f", info_table$scale)
write.table(
  info_table,
  sep = " & ",
  eol = "\\\\\n",
  quote = FALSE,
  row.names = FALSE
)

#-------------------------------------------------------------------------------
# generate test data

training_data <- lapply(
  seq_len(n_sim),
  function(k) generate_data(
    data = covariate_data,
    n_obs_intv = n_obs_intv,
    n_obs_observ = n_obs_observ,
    residuals = res,
    beta = beta,
    gamma = gamma
  )
)
training_data <- lapply(
  training_data,
  function(x) x[!x$intervention %in% hidden_confounders, ]
)
test_data <- lapply(
  seq_len(n_sim),
  function(k) generate_data(
    data = covariate_data,
    n_obs_intv = n_obs_intv,
    n_obs_observ = n_obs_intv,
    residuals = res,
    beta = beta,
    gamma = gamma
  )
)

for (i in seq_len(n_sim)) {
  training_data[[i]]$sim <- i
  test_data[[i]]$sim <- i
}

training_data <- do.call(rbind, training_data)
test_data <- do.call(rbind, test_data)

write.table(
  training_data,
  "data/simulation_2_training_data.txt",
  sep = ";",
  row.names = FALSE
)
write.table(
  test_data,
  "data/simulation_2_test_data.txt",
  sep = ";",
  row.names = FALSE
)
