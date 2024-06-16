#-----------------------------------------------------------------------------------------------------------------------
# modules

import os
import cputils as ut
import numpy as np
import pandas as pd

#-----------------------------------------------------------------------------------------------------------------------
# parameters

## all parameters
all_pars = pd.DataFrame({'env_ind': np.arange(50)})
all_pars = all_pars.merge(
  pd.DataFrame({'training_type' : ['observational', 'pooled']}),
  how = 'cross'
)
all_pars = all_pars.merge(
  pd.DataFrame({'rho' : [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]}),
  how = 'cross'
)
all_pars = all_pars.reset_index(drop = True)

## get index
i = int(os.environ['SLURM_ARRAY_TASK_ID'])

## environment to make predictions for (0:49)
env_ind = all_pars['env_ind'].iloc[i]

## pooled or observational data
training_type = all_pars['training_type'].iloc[i]

## type I error (1 - confidence level; only this alpha)
alpha = 0.1

## robustness parameter, rho = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
rho = all_pars['rho'].iloc[i]

## seed to reproduce split
this_seed = 20230531 + env_ind

#-----------------------------------------------------------------------------------------------------------------------
# read and prepare data

## read training, test data
if training_type == 'observational':
    train = pd.read_csv('data/cp_training_obs.csv', sep=';')
elif training_type == 'pooled':
  train = pd.read_csv('data/cp_training_pooled.csv', sep = ';')
test = pd.read_csv('data/cp_test.csv', sep=';')

## select environment
env = test.iloc[:,0]
uenv = env.unique()
e = uenv[env_ind]

## train and test data
test = test.iloc[:,1:]
train = train.values
test = test.values[env == e,:]

## get coverage, interval width
out = ut.fit_pred_cp(train, test, rho, alpha, this_seed)
out['env'] = e

## export results
out.to_csv(
    'cp_results_' + str(i) + '.csv',
    index = False,
    sep = ';'
)
