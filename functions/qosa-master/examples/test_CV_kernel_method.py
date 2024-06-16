# -*- coding: utf-8 -*-

import numba as nb
import numpy as np
import pandas as pd
from tqdm import tqdm
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

n_sample = 10**4

# First sample to compute the conditional CDF
X = np.random.exponential(size = (n_sample, 2))
Y1 = X[:,0] - X[:,1]
X1 = X[:,0]

# Second sample to estimate the index
X = np.random.exponential(size = (n_sample, 2))
Y2 = X[:,0] - X[:,1]
X2 = X[:,0]
   
alpha = np.array([0.3, 0.5, 0.7])
#bandwidth = np.unique(np.logspace(np.log10(0.002), np.log10(0.40), 60, endpoint=True, dtype=np.float64))
bandwidth = np.linspace(0.0009, 1., 50, endpoint=True, dtype=np.float64)

numerator = np.zeros((bandwidth.shape[0], alpha.shape[0]))

estimator = UnivariateQuantileRegressionKernel()
estimator.fit(X2, Y2)

for i in tqdm(range(bandwidth.shape[0])):
    conditional_quantiles = estimator.predict(X=X1,
                                              alpha=alpha,
                                              bandwidth=bandwidth[i])

    numerator[i, :] = _averaged_check_function_alpha_array(Y1.reshape(-1,1)-conditional_quantiles,
                                                           alpha)

index_columns = [alpha]
index_row = [bandwidth]
df_numerator = pd.DataFrame(index = index_row, columns = index_columns)
df_numerator[:] = numerator
df_numerator = df_numerator.rename_axis(index='bandwidth', columns="alpha")

df_numerator.to_csv('./Ibrahima_numerator_CTE.csv')