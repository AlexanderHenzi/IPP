# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm


from qosa import (MinimumBasedQosaIndices, qosa_Min__Min_in_Leaves, qosa_Min__Weighted_Min,
                  QuantileBasedQosaIndices, qosa_Quantile__Kernel_CDF)


# ------------------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ------------------------------------------------------------------------------------
#
# Compute the QOSA indices with two methods based on minimum and random forests as
# well as the Kernel estimator developped by Veronique and Ibrahima
#
# ------------------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ------------------------------------------------------------------------------------

seed = 888
rng = np.random.default_rng(seed)
dim = 2
n_RMSE = 10**2
n_samples = 10**4
n_trees = 10**2
min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(300), 20, endpoint=True, dtype=int))
n_min_samples_leaf = min_samples_leaf.shape[0]
bandwidth = np.unique(np.logspace(np.log10(0.001), np.log10(1.), 20, endpoint=True, dtype=np.float64))
n_bandwidth = bandwidth.shape[0]
alpha = np.array([0.1, 0.25, 0.5, 0.75, 0.99])
n_alphas = alpha.shape[0]
scale_normal = 0.5 # std of the normal random variable used for the additional noise

# Weighted Min: outer expectation computed with an aditional sample
method_1 = qosa_Min__Weighted_Min(alpha=alpha,
                                  n_estimators=n_trees,
                                  min_samples_leaf=min_samples_leaf,
                                  used_bootstrap_samples=False,
                                  optim_by_CV=True,
                                  n_fold=3)

# Min in leaves: outer expectation computed within the tree
method_2 = qosa_Min__Min_in_Leaves(alpha=alpha,
                                   n_estimators=n_trees,
                                   min_samples_leaf=min_samples_leaf,
                                   used_bootstrap_samples=False,
                                   optim_by_CV=True,
                                   n_fold=3)

# Kernel CDF: estimator developped by Veronique and Ibrahima
method_3 = qosa_Quantile__Kernel_CDF(alpha=alpha,
                                   bandwidth=bandwidth,
                                   optim_by_CV=True,
                                   n_fold=3)

results = [np.empty((n_RMSE, n_alphas, dim), dtype=np.float64) for i in range(3)]
for i in tqdm(range(n_RMSE)):
    X1 = rng.exponential(size = (n_samples, 2))
    Y1 = X1[:,0] - X1[:,1] + rng.normal(scale=scale_normal, size=n_samples)

    # Second sample to estimate the outer expectation of the index
    X2 = rng.exponential(size = (n_samples, 2))
    Y2 = X2[:,0] - X2[:,1] + rng.normal(scale=scale_normal, size=n_samples)
        
    qosa_1 = MinimumBasedQosaIndices()
    qosa_1.feed_sample(X1, Y1, "Weighted_Min", X2)
    qosa_results_1 = qosa_1.compute_indices(method=method_1)
    results[0][i,:,:] = qosa_results_1.qosa_indices_estimates
    
    # qosa_2 = MinimumBasedQosaIndices()
    # qosa_2.feed_sample(X1, Y1, "Min_in_Leaves")
    # qosa_results_2 = qosa_2.compute_indices(method=method_2)
    # results[1][i,:,:] = qosa_results_2.qosa_indices_estimates[0]
    
    # qosa_3 = QuantileBasedQosaIndices()
    # qosa_3.feed_sample(X1, Y1, X2, Y2)
    # qosa_results_3 = qosa_3.compute_indices(method=method_3)
    # results[2][i,:,:] = qosa_results_3.qosa_indices_estimates


# -----------------------------------------------------------------------------
# Export the results
# -----------------------------------------------------------------------------

exported_result = pd.DataFrame(index=range(n_RMSE),
                               columns=[x + y for x in ['Q1_', 'Q2_', 'S_'] for y in ['X1', 'X2']])
for i, alpha_temp in enumerate(alpha):
    for j in range(3):
        for k in range(dim):
            exported_result.iloc[:,j*2+k] = results[j][:,i,k]
    
    output_path = ('./n_RMSE_%d_n_samples_%d_n_min_samples_leaf_%d_n_trees_%d_n_bandwidth_%d_alpha_%s.csv' % (
        n_RMSE, n_samples, n_min_samples_leaf, n_trees, n_bandwidth, str(alpha_temp).replace('.', '_')))
    exported_result.to_csv(output_path)

# ----------------------------------------------------------------------------
# Compute the RMSE for Q1 estimator with theoretical values given by Véronique
# ----------------------------------------------------------------------------

theoretical_values = np.array([[0.11247, 0.16821, 0.259766, 0.369992, 0.605872],
                               [0.468963, 0.369743, 0.2594, 0.167689, 0.0609274]])

np.sqrt(((results[0][:,:,0] - theoretical_values[0,:])**2).mean(axis=0))
np.sqrt(((results[0][:,:,1] - theoretical_values[1,:])**2).mean(axis=0))
