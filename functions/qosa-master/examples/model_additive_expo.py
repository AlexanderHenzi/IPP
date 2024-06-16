# -*- coding: utf-8 -*-


import numpy as np 


# -----------------------------------------------------------------------------
#
# Design of the Additive Exponential model
#
# -----------------------------------------------------------------------------

from qosa.tests import AdditiveExponential

model = AdditiveExponential()


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
method = 'Forest_1'
qosa.build_sample(model=model, n_samples=n_samples, estimator=method)

alpha = np.array([0.2, 0.4, 0.6, 0.8])
model.alpha = alpha # in order to compute the true qosa indices
n_estimators = 10
min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(1500), 20, endpoint=True, dtype=int))

# With optim
qosa_results = qosa.compute_indices(alpha, n_estimators=n_estimators, 
                                    min_samples_leaf=min_samples_leaf, optim_by_CV=True, n_fold=3)
print('Forest_1 with optim')
qosa_results.qosa_indices
qosa_results.min_samples_leaf_by_dim_and_alpha

# Without optim
qosa_results = qosa.compute_indices(alpha, n_estimators=n_estimators, optim_by_CV=False)
print('Forest_1 without optim')
qosa_results.qosa_indices

# ----------------
# By computing the argmin inside each tree
# ----------------
method = 'Forest_2'
qosa.build_sample(model=model, n_samples=n_samples, estimator=method)

alpha = np.array([0.2, 0.4, 0.6, 0.8])
model.alpha = alpha # in order to compute the true qosa indices
n_estimators = 10
min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(1500), 20, endpoint=True, dtype=int))

# With optim
qosa_results = qosa.compute_indices(alpha, n_estimators=n_estimators, 
                                    min_samples_leaf=min_samples_leaf, optim_by_CV=True, n_fold=3)
print('Forest_2 with optim')
qosa_results.qosa_indices
qosa_results.min_samples_leaf_by_dim_and_alpha

# Without optim
qosa_results = qosa.compute_indices(alpha, n_estimators=n_estimators, optim_by_CV=False)
print('Forest_2 without optim')
qosa_results.qosa_indices