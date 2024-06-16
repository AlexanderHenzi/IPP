# -*- coding: utf-8 -*-


import numpy as np 
import time

# -----------------------------------------------------------------------------
#
# Design of the Toy insurance model
#
# -----------------------------------------------------------------------------

from qosa.tests import ToyInsurance

GPD_params = [1.5, 0.25]
LN_params = [1.1, 0.6, 0.]
Gamma_params = [2., 0.6, 0.]

model = ToyInsurance(GPD_params=GPD_params, LN_params=LN_params, Gamma_params=Gamma_params)


# -----------------------------------------------------------------------------
#
# Compute the qosa indices
#
# -----------------------------------------------------------------------------

from qosa import QosaIndices

qosa = QosaIndices(model.input_distribution)

n_samples = 10**4

# ----------------
# By plugging the conditional quantile computed with the random forest method
# ----------------
estimation_method = 'compute_quantile'
qosa.build_sample(model=model, n_samples=n_samples, estimation_method=estimation_method)

alpha = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
model.alpha = alpha # in order to compute the true qosa indices
n_estimators = 10**1
min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(1500), 10, endpoint=True, dtype=int))

# With optim
quantile_method = 'Averaged_Quantile_OOB'
start = time.time()
qosa_results = qosa.compute_indices(alpha, 
									quantile_method = quantile_method,
									n_estimators=n_estimators, 
                                    min_samples_leaf=min_samples_leaf,
                                    optim_by_CV=True, 
                                    n_fold=3,
                                    used_bootstrap_samples=False)
print("Total time = ",time.time() - start)
print('Forest_1 with optim')
qosa_results.qosa_indices
qosa_results.min_samples_leaf_by_dim_and_alpha

## Without optim
quantile_method = 'Weighted_CDF'
qosa_results = qosa.compute_indices(alpha, 
									quantile_method=quantile_method,
									n_estimators=n_estimators,
									optim_by_CV=False,
									used_bootstrap_samples=False)
print('Forest_1 without optim')
qosa_results.qosa_indices

# ----------------
# By computing the argmin inside each tree
# ----------------
estimation_method = 'compute_mean'
qosa.build_sample(model=model, n_samples=n_samples, estimation_method=estimation_method)

alpha = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
model.alpha = alpha # in order to compute the true qosa indices
n_estimators = 10**1
min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(1500), 10, endpoint=True, dtype=int))

# With optim
start = time.time()
qosa_results = qosa.compute_indices(alpha, 
									n_estimators=n_estimators, 
                                   	min_samples_leaf=min_samples_leaf,
                                   	optim_by_CV=True,
                                   	n_fold=3,
                                   	used_bootstrap_samples=False)
print("Total time = ",time.time() - start)
print('Forest_2 with optim')
qosa_results.qosa_indices
qosa_results.min_samples_leaf_by_dim_and_alpha

# Without optim
qosa_results = qosa.compute_indices(alpha, 
									n_estimators=n_estimators,
									optim_by_CV=False,
									used_bootstrap_samples=False)
print('Forest_2 without optim')
qosa_results.qosa_indices