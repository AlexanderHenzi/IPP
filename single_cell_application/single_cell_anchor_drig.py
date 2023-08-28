'''
Refer to https://github.com/xwshen51/DRIG/blob/main/single_cell.ipynb
'''


# packages
import pandas as pd
import os 
from functions.anchor_drig import *
import pandas as pd

# load data
df = pd.read_csv("data/dataset_rpe1.csv")
interv_label = df['interventions']
groups = df.groupby(['interventions'])
## centralize data
df.iloc[:,:-1] = df.iloc[:,:-1] - groups.get_group('non-targeting').iloc[:,:-1].mean()
## observational data
data_obs = groups.get_group('non-targeting').iloc[:,:-1].to_numpy()
## interventional data
interv_obs = df.columns[:-1].to_numpy()
interv_hidden = set(interv_label) - set(interv_obs) - set(['non-targeting'])
data_interv_obs = []
data_interv_hidden = []
for interv_i in interv_obs:
    data_interv_obs.append(groups.get_group(interv_i).iloc[:,:-1].to_numpy())
for interv_i in interv_hidden:
    data_interv_hidden.append(groups.get_group(interv_i).iloc[:,:-1].to_numpy())
data_interv_obs = dict(zip(interv_obs, data_interv_obs))
data_interv_hidden = dict(zip(interv_hidden, data_interv_hidden))
train_data = [data_obs] + list(data_interv_obs.values())

# get all test environments
test_mse_ols_obs = np.array(test_mse_list(list(data_interv_hidden.values()), est(train_data, method='ols_obs')))
env_idx = test_mse_ols_obs.argsort()[::-1][:412]
data_test_large_shift = [list(data_interv_hidden.values())[i] for i in env_idx]
data_test_large_shift_gene = [list(data_interv_hidden.keys())[i] for i in env_idx]

# estimate anchor regressoin and drig, compute error
gammas = np.arange(0, 50, 1)
num_gammas = len(gammas)
num_envs = len(data_test_large_shift)
results = pd.DataFrame(columns=['method', 'gamma', 'interv_gene', 'test_mse'])
mses_drig = []
mses_anc = []
for i in range(num_gammas):
    mses_drig = test_mse_list(data_test_large_shift, est(train_data, method='drig', gamma=gammas[i]))
    mses_anc = test_mse_list(data_test_large_shift, est(train_data, method='anchor', gamma=gammas[i]))
    result = pd.DataFrame({
        'method': np.concatenate([np.repeat('DRIG', num_envs), np.repeat('anchor regression', num_envs)], axis=0),
        'gamma': np.repeat(gammas[i], num_envs*2),
        'interv_gene': data_test_large_shift_gene*2,
        'test_mse': np.concatenate([mses_drig, mses_anc], axis=0)
    })
    results = result if results is None else pd.concat([results, result], ignore_index=True)
    
# export results
results.to_csv('results/anchor_drig.csv')

