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

model = ToyInsurance()

alpha = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
model.alpha = alpha

ot.RandomGenerator.SetSeed(0)
np.random.seed(0)

# -----------------------------------------------------------------------------
# Computing the RMSE for the qosa indices without CV
# -----------------------------------------------------------------------------

set_style_paper()

parameters = {'model': model,
              'alpha': alpha,
              'n_samples': 10**3,
              'method': 'Forest_1',
              'n_estimators': 10**1,
              'optim_by_CV': False,
              'n_fold': 3,
              'min_samples_leaf_start': 5,
              'min_samples_leaf_stop': 1500,
              'min_samples_leaf_num': 5,
              'n_RMSE': 5}
compute_and_plot_RMSE_results(**parameters)

parameters = {'model': model,
              'alpha': alpha,
              'n_samples': 10**3,
              'method': 'Forest_2',
              'n_estimators': 10**1,
              'optim_by_CV': False,
              'n_fold': 3,
              'min_samples_leaf_start': 5,
              'min_samples_leaf_stop': 1500,
              'min_samples_leaf_num': 5,
              'n_RMSE': 5}
compute_and_plot_RMSE_results(**parameters)