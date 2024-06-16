# -*- coding: utf-8 -*-

import numpy as np
import openturns as ot

from qosa.tests import AdditiveExponential
from qosa.plots import compute_and_plot_RMSE_results, set_style_paper


# -----------------------------------------------------------------------------
#
# Design of the Additive Exponential model
#
# -----------------------------------------------------------------------------

lambda_params = [0.5, 1., 1.5, 2.]
model = AdditiveExponential(lambda_params=lambda_params)

alpha = np.asarray([0.1, 0.3, 0.7, 0.99])
model.alpha = alpha

ot.RandomGenerator.SetSeed(0)
np.random.seed(0)

# -----------------------------------------------------------------------------
# Computing the RMSE for the qosa indices with CV
# -----------------------------------------------------------------------------

set_style_paper()

# Forest 1

# parameters = {'model': model,
#               'alpha': alpha,
#               'n_samples': 10**3,
#               'method': 'Forest_1',
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
#               'method': 'Forest_1',
#               'n_estimators': 10**1,
#               'optim_by_CV': True,
#               'n_fold': 3,
#               'used_bootstrap_samples': True,
#               'min_samples_leaf_start': 5,
#               'min_samples_leaf_stop': 1500,
#               'min_samples_leaf_num': 5,
#               'n_RMSE': 2}
# compute_and_plot_RMSE_results(**parameters)

# Forest 2

parameters = {'model': model,
              'alpha': alpha,
              'n_samples': 5*10**3,
              'method': 'Forest_2',
              'n_estimators': 3*10**2,
              'optim_by_CV': True,
              'n_fold': 3,
              'used_bootstrap_samples': False,
              'min_samples_leaf_start': 5,
              'min_samples_leaf_stop': 1500,
              'min_samples_leaf_num': 20,
              'n_RMSE': 20}
compute_and_plot_RMSE_results(**parameters)

parameters = {'model': model,
              'alpha': alpha,
              'n_samples': 5*10**3,
              'method': 'Forest_2',
              'n_estimators': 3*10**2,
              'optim_by_CV': True,
              'n_fold': 3,
              'used_bootstrap_samples': True,
              'min_samples_leaf_start': 5,
              'min_samples_leaf_stop': 1500,
              'min_samples_leaf_num': 20,
              'n_RMSE': 20}
compute_and_plot_RMSE_results(**parameters)