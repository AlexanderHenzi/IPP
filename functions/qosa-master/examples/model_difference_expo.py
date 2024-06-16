# -*- coding: utf-8 -*-

import time
import numpy as np 


# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------
#
# Design of the Difference Exponential model
#
# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------

from qosa.tests import BiDifferenceExponential

lambda_param = 1
model = BiDifferenceExponential(lambda_param)


# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------
#
# Compute the qosa indices
#
# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# By plugging the conditional quantile computed with the random forest method
# -----------------------------------------------------------------------------

from qosa import QuantileBasedQosaIndices

qosa = QuantileBasedQosaIndices()
n_samples = 10**4
qosa.build_sample(model=model, n_samples=n_samples)

from qosa import qosa_Quantile__Averaged_Quantile, qosa_Quantile__Weighted_CDF

alpha = np.array([0.1, 0.5, 0.7, 0.9])
n_estimators = 10**1
min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(500), 10, endpoint=True, dtype=int))

############
# With optim
############

# 1 #
method = qosa_Quantile__Averaged_Quantile(alpha=alpha,
                                          n_estimators=n_estimators,
                                          min_samples_leaf=min_samples_leaf,
                                          used_bootstrap_samples=False,
                                          optim_by_CV=True,
                                          CV_strategy="OOB",
                                          n_fold=3)

start = time.perf_counter()
qosa_results = qosa.compute_indices(method)
print("Total time = ",time.perf_counter() - start)

qosa_results.qosa_indices_estimates
qosa_results.optimal_parameter_by_CV
qosa_results.true_qosa_indices

# 2 #
method = qosa_Quantile__Weighted_CDF(alpha=alpha,
                                     n_estimators=n_estimators,
                                     min_samples_leaf=min_samples_leaf,
                                     used_bootstrap_samples=False,
                                     optim_by_CV=True,
                                     CV_strategy="OOB",
                                     n_fold=3)

start = time.perf_counter()
qosa_results = qosa.compute_indices(method)
print("Total time = ",time.perf_counter() - start)

qosa_results.qosa_indices_estimates
qosa_results.optimal_parameter_by_CV
qosa_results.true_qosa_indices

###############
# Without optim
###############

# 1 #
method = qosa_Quantile__Averaged_Quantile(alpha=alpha,
                                n_estimators=n_estimators,
                                min_samples_leaf=20,
                                used_bootstrap_samples=False,
                                optim_by_CV=False)

start = time.perf_counter()
qosa_results = qosa.compute_indices(method)
print("Total time = ",time.perf_counter() - start)

qosa_results.qosa_indices_estimates
qosa_results.optimal_parameter_by_CV
qosa_results.true_qosa_indices

# 2 #
method = qosa_Quantile__Weighted_CDF(alpha=alpha,
                           n_estimators=n_estimators,
                           min_samples_leaf=20,
                           used_bootstrap_samples=True,
                           optim_by_CV=False)

start = time.perf_counter()
qosa_results = qosa.compute_indices(method)
print("Total time = ",time.perf_counter() - start)

qosa_results.qosa_indices_estimates
qosa_results.optimal_parameter_by_CV
qosa_results.true_qosa_indices

# -----------------------------------------------------------------------------
# By plugging the conditional quantile computed with the kernel method
# -----------------------------------------------------------------------------

from qosa import QuantileBasedQosaIndices

qosa = QuantileBasedQosaIndices()
n_samples = 10**4
qosa.build_sample(model=model, n_samples=n_samples)

from qosa import qosa_Quantile__Kernel_CDF

alpha = np.array([0.1, 0.5, 0.7, 0.9])
bandwidth = np.linspace(0.009, 0.8, 10, endpoint=True, dtype=np.float64)
bandwidth = np.unique(np.logspace(np.log10(0.001), np.log10(1.), 20, endpoint=True, dtype=np.float64))


############
# With optim
############

method = qosa_Quantile__Kernel_CDF(alpha=alpha,
                         bandwidth=bandwidth,
                         optim_by_CV=True,
                         n_fold=3)

start = time.perf_counter()
qosa_results = qosa.compute_indices(method)
print("Total time = ",time.perf_counter() - start)

qosa_results.qosa_indices_estimates
qosa_results.optimal_parameter_by_CV
qosa_results.true_qosa_indices

###############
# Without optim
###############

method = qosa_Quantile__Kernel_CDF(alpha=alpha,
                             bandwidth=None,
                             optim_by_CV=False,
                             n_fold=3)

start = time.perf_counter()
qosa_results = qosa.compute_indices(method)
print("Total time = ",time.perf_counter() - start)

qosa_results.qosa_indices_estimates
qosa_results.optimal_parameter_by_CV
qosa_results.true_qosa_indices

# -------------------------------------------------------------------------------
# By computing the argmin inside each tree computed with the random forest method
# -------------------------------------------------------------------------------

from qosa import MinimumBasedQosaIndices

qosa = MinimumBasedQosaIndices()
n_samples = 10**3
qosa.build_sample(model=model, n_samples=n_samples, method='Kernel_Min')

from qosa import qosa_Min__Min_in_Leaves, qosa_Min__Weighted_Min

alpha = np.array([0.1, 0.5, 0.7, 0.9])
n_estimators = 10**1
min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(500), 10, endpoint=True, dtype=int))

############
# With optim
############

# 1 #
method = qosa_Min__Min_in_Leaves(alpha=alpha,
                                 n_estimators=n_estimators,
                                 min_samples_leaf=min_samples_leaf,
                                 used_bootstrap_samples=True,
                                 optim_by_CV=True,
                                 n_fold=3)

start = time.perf_counter()
qosa_results = qosa.compute_indices(method)
print("Total time = ",time.perf_counter() - start)

qosa_results.qosa_indices_estimates
qosa_results.optimal_parameter_by_CV
qosa_results.true_qosa_indices

# 2 #
method = qosa_Min__Weighted_Min(alpha=alpha,
                                n_estimators=n_estimators,
                                min_samples_leaf=min_samples_leaf,
                                used_bootstrap_samples=False,
                                optim_by_CV=True,
                                n_fold=3)

start = time.perf_counter()
qosa_results = qosa.compute_indices(method)
print("Total time = ",time.perf_counter() - start)

qosa_results.qosa_indices_estimates
qosa_results.optimal_parameter_by_CV
qosa_results.true_qosa_indices

###############
# Without optim
###############

# 1 #
method = qosa_Min__Min_in_Leaves(alpha=alpha,
                                 n_estimators=n_estimators,
                                 min_samples_leaf=20,
                                 used_bootstrap_samples=False,
                                 optim_by_CV=False)

start = time.perf_counter()
qosa_results = qosa.compute_indices(method)
print("Total time = ",time.perf_counter() - start)

qosa_results.qosa_indices_estimates
qosa_results.optimal_parameter_by_CV
qosa_results.true_qosa_indices

# 2 #
method = qosa_Min__Weighted_Min(alpha=alpha,
                                n_estimators=n_estimators,
                                min_samples_leaf=20,
                                used_bootstrap_samples=False,
                                optim_by_CV=False)

start = time.perf_counter()
qosa_results = qosa.compute_indices(method)
print("Total time = ",time.perf_counter() - start)

qosa_results.qosa_indices_estimates
qosa_results.optimal_parameter_by_CV
qosa_results.true_qosa_indices

# ------------------------------------------------------------------------
# By computing the argmin inside each tree computed with the kernel method
# ------------------------------------------------------------------------

from qosa import MinimumBasedQosaIndices

qosa = MinimumBasedQosaIndices()
n_samples = 10**3
qosa.build_sample(model=model, n_samples=n_samples, method='Kernel_Min')

from qosa import qosa_Min__Kernel_Min

alpha = np.array([0.1, 0.5, 0.7, 0.9])
bandwidth = np.linspace(0.009, 0.8, 20, endpoint=True, dtype=np.float64)

############
# With optim
############

method = qosa_Min__Kernel_Min(alpha=alpha,
                              bandwidth=bandwidth,
                              optim_by_CV=True,
                              n_fold=3)

start = time.perf_counter()
qosa_results = qosa.compute_indices(method)
print("Total time = ",time.perf_counter() - start)

qosa_results.qosa_indices_estimates
qosa_results.optimal_parameter_by_CV
qosa_results.true_qosa_indices

###############
# Without optim
###############

method = qosa_Min__Kernel_Min(alpha=alpha,
                              bandwidth=None,
                              optim_by_CV=False)

start = time.perf_counter()
qosa_results = qosa.compute_indices(method)
print("Total time = ",time.perf_counter() - start)

qosa_results.qosa_indices_estimates
qosa_results.optimal_parameter_by_CV
qosa_results.true_qosa_indices
