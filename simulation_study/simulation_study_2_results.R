# --- global settings ----------------------------------------------------------

# packages
library(ggplot2)
library(dplyr)

# settings
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

# data generation parameters
response_gene <- "ENSG00000173812"
hidden_confounders <- c(
  "ENSG00000187514",
  "ENSG00000075624",
  "ENSG00000147604",
  "ENSG00000110700" 
)

# --- read data ----------------------------------------------------------------

fl <- list.files("data/simulation_2_results/")
data <- vector("list", length(fl))
for (i in seq_along(fl)) {
  data[[i]] <- read.csv(paste0("data/simulation_2_results/", fl[i]), sep = ";")
}
data <- do.call(rbind, data)
data <- data %>% 
  mutate(
    method = factor(
      method,
      labels = c(
        "IPP (LogS)",
        "IPP (SCRPS)",
        "IPP (CRPS)",
        "DRIG",
        "Anchor regression",
        "Conformal",
        "Conformal (weighted)"
      ),
      levels = c(
        "ipp_logs",
        "ipp_scrps",
        "ipp_crps",
        "drig",
        "anchor",
        "rcp",
        "dwrcp"
      ),
      ordered = TRUE
    ),
    score = factor(
      score,
      levels = c("logs", "scrps", "crps", "mse", "in_pi", "pi_width"),
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


# --- analysis of hidden confounder interventions ------------------------------

# average scores
hidden <- data[data$env %in% hidden_confounders, ]
hidden_summary <- hidden %>%
  group_by(method, penalty, score, env) %>%
  summarise(
    mean = mean(value),
    median = median(value),
    mplusse = mean(value) + sd(value) / sqrt(length(value)),
    mminusse = mean(value) - sd(value) / sqrt(length(value)),
  )

# figure
scores <- as.character(sort(unique(hidden_summary$score)))
plots <- vector("list", length(scores))
for (i in seq_along(scores)) {
  lps <- ifelse(scores[i] == "Length", "bottom", "none")
  xlb <- if (scores[i] == "Length") "Penalty" else element_blank()
  
  plots[[i]] <- hidden_summary %>%
    filter(score == scores[i]) %>%
    ggplot() +
    geom_line(aes(x = penalty, y = mean, group = env, color = env)) +
    geom_ribbon(
      aes(x = penalty, ymin = mminusse, ymax = mplusse, group = env, fill = env),
      alpha = 0.25
    ) +
    facet_grid(cols = vars(method), scales = "free") +
    theme(legend.position = lps, axis.text.x = element_text(size = 6)) +
    scale_color_manual(values = colpal[seq_len(4)]) +
    scale_fill_manual(values = colpal[seq_len(4)]) +
    labs(
      y = scores[i],
      x = xlb,
      color = element_blank(),
      fill = element_blank()
    )
  
  if (scores[i] == "Coverage") {
    plots[[i]] <- plots[[i]] +
      geom_hline(
        data = tibble(yintercept = 0.9, score = "Coverage"),
        aes(yintercept = yintercept),
        col = "red"
      )
  }
}

figure <- ggpubr::ggarrange(
  plotlist = plots,
  ncol = 1,
  heights = c(1, 1, 1, 1, 1, 1.5)
)

pdf("temporary_files/simulation2hidden.pdf", width = 8, height = 10)
print(figure)
dev.off()

# --- analysis of interventions on observed ------------------------------------

# average scores
obs <- data[!data$env %in% hidden_confounders, ]
obs_summary <- obs %>%
  group_by(method, penalty, score, env) %>%
  summarise(
    mean = mean(value),
    median = median(value),
    mplusse = mean(value) + sd(value) / sqrt(length(value)),
    mminusse = mean(value) - sd(value) / sqrt(length(value)),
  ) %>%
  mutate(env = replace(env, which(env == "non-targeting"), "observational"))

# figure
scores <- as.character(sort(unique(obs_summary$score)))
plots <- vector("list", length(scores))
for (i in seq_along(scores)) {
  lps <- ifelse(scores[i] == "Length", "bottom", "none")
  xlb <- if (scores[i] == "Length") "Penalty" else element_blank()
  
  plots[[i]] <- obs_summary %>%
    filter(score == scores[i]) %>%
    ggplot() +
    geom_line(aes(x = penalty, y = mean, group = env, color = env)) +
    geom_ribbon(
      aes(x = penalty, ymin = mminusse, ymax = mplusse, group = env, fill = env),
      alpha = 0.25
    ) +
    facet_grid(cols = vars(method), scales = "free") +
    theme(legend.position = lps, axis.text.x = element_text(size = 6)) +
    scale_color_manual(values = colpal[seq_len(6)]) +
    scale_fill_manual(values = colpal[seq_len(6)]) +
    labs(
      y = scores[i],
      x = xlb,
      color = element_blank(),
      fill = element_blank()
    )
  
  if (scores[i] == "Coverage") {
    plots[[i]] <- plots[[i]] +
      geom_hline(
        data = tibble(yintercept = 0.9, score = "Coverage"),
        aes(yintercept = yintercept),
        col = "red"
      )
  }
}

figure <- ggpubr::ggarrange(
  plotlist = plots,
  ncol = 1,
  heights = c(1, 1, 1, 1, 1, 1.5)
)

pdf("temporary_files/simulation2obs.pdf", width = 8, height = 10)
print(figure)
dev.off()

