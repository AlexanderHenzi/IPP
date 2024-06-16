# -*- coding: utf-8 -*-

import numpy as np
import openturns as ot

from qosa.plots import compute_and_plot_RMSE_results, set_style_paper
from qosa.tests import DifferenceExponential


# -----------------------------------------------------------------------------
#
# Design of the Difference Exponential model
#
# -----------------------------------------------------------------------------

lambda_param = 1
model = DifferenceExponential(lambda_param)

ot.RandomGenerator.SetSeed(0)
np.random.seed(0)

# -----------------------------------------------------------------------------
# Computing the RMSE for the qosa indices with CV
# -----------------------------------------------------------------------------

set_style_paper()

# Compute quantile

from qosa import Kernel_CDF, Weighted_CDF

alpha = np.asarray([0.01, 0.3, 0.7, 0.99])
optim_by_CV = True
n_fold = 4

#n_estimators = 10**1
#min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(1500), 20, endpoint=True, dtype=int)) # Size of the leaf nodes to try
#method = Weighted_CDF(alpha=alpha,
#                      n_estimators=n_estimators,
#                      min_samples_leaf=min_samples_leaf,
#                      optim_by_CV=optim_by_CV,
#                      n_fold=n_fold)

#bandwidth = np.linspace(0.01, 0.90, 20, endpoint=True, dtype=np.float64)
bandwidth = np.unique(np.logspace(np.log10(0.01), np.log10(0.90), 40, endpoint=True, dtype=np.float64))

method = Kernel_CDF(alpha=alpha,
                    bandwidth=bandwidth,
                    optim_by_CV=optim_by_CV,
                    n_fold=n_fold)

parameters = {'model': model,
              'method': method,
              'n_samples': 4*10**4,
              'n_RMSE': 100,
              'parameters_for_no_CV': None}

compute_and_plot_RMSE_results(**parameters)

#parameters = {'model': model,
#              'alpha': alpha,
#              'n_samples': 10**4,
#              'method': 'Forest_1',
#              'n_estimators': 3*10**2,
#              'optim_by_CV': True,
#              'n_fold': 3,
#              'used_bootstrap_samples': True,
#              'min_samples_leaf_start': 5,
#              'min_samples_leaf_stop': 1500,
#              'min_samples_leaf_num': 20,
#              'n_RMSE': 20}
#compute_and_plot_RMSE_results(**parameters)

# Forest 2

# parameters = {'model': model,
#               'alpha': alpha,
#               'n_samples': 10**3,
#               'method': 'Forest_2',
#               'n_estimators': 10**1,
#               'optim_by_CV': True,
#               'n_fold': 3,
#               'used_bootstrap_samples': False,
#               'min_samples_leaf_start': 5,
#               'min_samples_leaf_stop': 1500,
#               'min_samples_leaf_num': 5,
#               'n_RMSE': 2}
# compute_and_plot_RMSE_results(**parameters)

# parameters = {'model': model,
#               'alpha': alpha,
#               'n_samples': 10**3,
#               'method': 'Forest_2',
#               'n_estimators': 10**1,
#               'optim_by_CV': True,
#               'n_fold': 3,
#               'used_bootstrap_samples': True,
#               'min_samples_leaf_start': 5,
#               'min_samples_leaf_stop': 1500,
#               'min_samples_leaf_num': 5,
#               'n_RMSE': 2}
# compute_and_plot_RMSE_results(**parameters)