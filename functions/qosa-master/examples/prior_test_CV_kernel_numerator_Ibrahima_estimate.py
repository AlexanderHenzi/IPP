# -*- coding: utf-8 -*-

import numba as nb
import numpy as np
import pandas as pd
from qosa import UnivariateQuantileRegressionKernel

@nb.njit("float64[:](float64[:,:], float64[:])", nogil=True, cache=False, parallel=True)
def _averaged_check_function_alpha_array(u, alpha):
    """
    Definition of the check function also called pinball loss function.
    """

    n_alphas = alpha.shape[0]
    n_samples = u.shape[0]

    check_function = np.empty((n_alphas, n_samples), dtype=np.float64)
    for i in nb.prange(n_samples):
        for j in range(n_alphas):
            check_function[j,i] = u[i,j]*(alpha[j] - (u[i,j] < 0.))

    averaged_check_function = np.empty((n_alphas), dtype=np.float64)
    for i in nb.prange(n_alphas):
        averaged_check_function[i] = check_function[i,:].mean()

    return averaged_check_function

n_sample = 2*10**4

# First sample to compute the conditional CDF
X = np.random.exponential(size = (n_sample, 2))
Y1 = X[:,0] - X[:,1]
X1 = X[:,0]

# Second sample to estimate the index
X = np.random.exponential(size = (n_sample, 2))
Y2 = X[:,0] - X[:,1]
X2 = X[:,0]

mean_Y = (Y1.mean() + Y2.mean())*0.5
    
alpha = np.array([0.3, 0.5, 0.7])
#bandwidth = np.unique(np.logspace(np.log10(0.002), np.log10(0.40), 60, endpoint=True, dtype=np.float64))
bandwidth = np.linspace(0.001, 0.10, 80, endpoint=True, dtype=np.float64)

numerator1 = np.zeros((bandwidth.shape[0], alpha.shape[0]))
numerator2 = np.zeros((bandwidth.shape[0], alpha.shape[0]))

estimator = UnivariateQuantileRegressionKernel()
estimator.fit(X2, Y2)

for i in range(bandwidth.shape[0]):
    print(i)
    conditional_quantiles = estimator.predict(X=X1,
                                              alpha=alpha,
                                              bandwidth=bandwidth[i])

    numerator1[i, :] = (Y1.reshape(-1, 1)*(Y1.reshape(-1, 1) > conditional_quantiles)).mean(axis=0) - mean_Y*(1-alpha)
    numerator2[i, :] = _averaged_check_function_alpha_array(Y1.reshape(-1,1)-conditional_quantiles,
                                                            alpha)

index_columns = [alpha]
index_row = [bandwidth]
df_numerator1 = pd.DataFrame(index = index_row, columns = index_columns)
df_numerator1[:] = numerator1

df_numerator2 = pd.DataFrame(index = index_row, columns = index_columns)
df_numerator2[:] = numerator2

df_numerator1.to_csv('./Ibrahima_numerator_CTE.csv')
df_numerator2.to_csv('./Ibrahima_numerator_check_function.csv')

#print('numerator1 = \n', numerator1)
#print('numerator2 = \n', numerator2)

#    theta_bis = np.percentile(Y_bis, q=alpha*100)
#    denominateur_bis = [p_temp*(Y*(Y>theta_temp)).mean() - meanY for p_temp, theta_temp in zip(p,theta_bis)]
#    
#    for j,n_min_leaf_temp in enumerate(n_min_leaf):
#        n_min_split_temp = n_min_split[j]
#               
#        quantForest = QuantileRegressionForest(n_estimators=n_tree, min_samples_split=n_min_split_temp, min_samples_leaf=n_min_leaf_temp, n_jobs=-1)
#        quantForest.fit(X1_bis, Y_bis)
#        conditional_quantiles = quantForest.predict(X1, alpha)
#        numerator = (Y[:,np.newaxis]*(Y[:,np.newaxis]>conditional_quantiles)).mean(axis=0)*p - meanY
#        print(numerator*(1-alpha))