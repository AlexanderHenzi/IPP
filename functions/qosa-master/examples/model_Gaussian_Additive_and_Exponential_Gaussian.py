# -*- coding: utf-8 -*-

import numpy as np
import time

from qosa import MinimumBasedQosaIndices, qosa_Min__Min_in_Leaves, QoseIndices
from qosa.tests import AdditiveGaussian, ExponentialGaussian


###########################
# Parameters of the model #
###########################

alpha = np.array([0.1, 0.4, 0.7, 0.95])
n_alphas = alpha.shape[0]
dim = 3
means = [0]*dim
std = [1, 1.4, 1.8]
beta = [1]*dim
corr = 0
correlation = [0, 0, corr]

model = AdditiveGaussian(dim=dim, means=means, std=std, beta=beta)
# model = ExponentialGaussian(dim=dim, means=means, std=std, beta=beta)
model.copula_parameters = correlation
model.alpha = alpha
first_order_qosa_indices, total_order_qosa_indices, qose_indices = model.qosa_indices # true indices


####################################
# Estimation of first QOSA indices #
####################################

qosa = MinimumBasedQosaIndices()
n_samples = 2*10**4
qosa.build_sample(model=model, n_samples=n_samples, method='Min_in_Leaves')

n_estimators = 10**1
min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(500), 10, endpoint=True, dtype=int))
method = qosa_Min__Min_in_Leaves(alpha=alpha,
                                 n_estimators=n_estimators,
                                 min_samples_leaf=min_samples_leaf,
                                 used_bootstrap_samples=False,
                                 optim_by_CV=True,
                                 n_fold=3)

start = time.perf_counter()
qosa_results = qosa.compute_indices(method)
print("Total time = ",time.perf_counter() - start)

qosa_indices = qosa_results.qosa_indices_estimates[1]
first_order_qosa_indices

##############################
# Estimation of QOSE indices #
##############################

qose = QoseIndices(model.input_distribution)
qose.build_sample(model=model, n_upsilon=10**5, n_perms=None, n_outer=2*10**4, n_inner=400)

start = time.perf_counter()
qose_results = qose.compute_indices(alpha, n_boot=1)
print("Total time = ",time.perf_counter() - start)

qose_results.qose_indices.T
qose_indices






