# -*- coding: utf-8 -*-

import numpy as np
import openturns as ot

from qosa.tests import ToyInsurance
from qosa.plots import compute_and_plot_RMSE_results, set_style_paper


# -----------------------------------------------------------------------------
#
# Design of the Toy insurance model
#
# -----------------------------------------------------------------------------

GPD_params = [1.5, 0.25]
LN_params = [1.1, 0.6, 0.]
Gamma_params = [2., 0.6, 0.]

model = ToyInsurance(GPD_params=GPD_params, LN_params=LN_params, Gamma_params=Gamma_params)

alpha = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
model.alpha = alpha

ot.RandomGenerator.SetSeed(0)
np.random.seed(0)

# -----------------------------------------------------------------------------
# Computing the RMSE for the qosa indices with CV
# -----------------------------------------------------------------------------

set_style_paper()

parameters = {'model': model,
             'alpha': alpha,
             'n_samples': 5*10**4,
             'method': 'Forest_1',
             'n_estimators': 3*10**2,
             'optim_by_CV': True,
             'n_fold': 5,
             'used_bootstrap_samples': True,
             'min_samples_leaf_start': 5,
             'min_samples_leaf_stop': 1500,
             'min_samples_leaf_num': 30,
             'n_RMSE': 50}
compute_and_plot_RMSE_results(**parameters)

# parameters = {'model': model,
#              'alpha': alpha,
#              'n_samples': 10**5,
#              'method': 'Forest_2',
#              'n_estimators': 3*10**2,
#              'optim_by_CV': True,
#              'n_fold': 5,
#              'min_samples_leaf_start': 5,
#              'min_samples_leaf_stop': 1500,
#              'min_samples_leaf_num': 20,
#              'n_RMSE': 10}
# compute_and_plot_RMSE_results(**parameters)
