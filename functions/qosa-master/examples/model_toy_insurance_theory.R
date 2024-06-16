library(evd)

param_GPD = c(1.5, 0.4)
param_LN = c(1, 0.7)
param_gamma = c(2.8, 0.7)
taille <- 10**8

GPD <- rgpd(taille, 0, param_GPD[1], param_GPD[2])
LN <- rlnorm(taille, param_LN[1], param_LN[2])
gamma <- rgamma(taille, param_gamma[1], param_gamma[2])

Y = GPD + LN + gamma
alpha = c(0.1, 0.3, 0.5, 0.7, 0.9) 

quantile_Y <- quantile(Y, alpha)
esp_Y <- param_GPD[1]/(1-param_GPD[2]) +
         exp(param_LN[1] + 0.5*param_LN[2]**2) +
         param_gamma[1]/param_gamma[2]

q_alpha_cond_1 <- quantile(LN + gamma, alpha)
q_alpha_cond_2 <- quantile(GPD + gamma, alpha)
q_alpha_cond_3 <- quantile(GPD + LN, alpha)

Esp_tronq<-rep(0, length(alpha))
Esp_tronq_1<-rep(0, length(alpha))
Esp_tronq_2<-rep(0, length(alpha))
Esp_tronq_3<-rep(0, length(alpha))

for(i in 1:length(alpha))
{
  Esp_tronq[i] <- mean(Y*(Y <= quantile_Y[i]))
  Esp_tronq_1[i] <- mean(Y*((LN + gamma) <= q_alpha_cond_1[i]))
  Esp_tronq_2[i] <- mean(Y*((GPD + gamma) <= q_alpha_cond_2[i]))
  Esp_tronq_3[i] <- mean(Y*((GPD + LN) <= q_alpha_cond_3[i]))
}

qosa_1 <- (Esp_tronq_1 - Esp_tronq)/(alpha*esp_Y - Esp_tronq)
qosa_2 <- (Esp_tronq_2 - Esp_tronq)/(alpha*esp_Y - Esp_tronq)
qosa_3 <- (Esp_tronq_3 - Esp_tronq)/(alpha*esp_Y - Esp_tronq)

