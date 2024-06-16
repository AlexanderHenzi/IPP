# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as ss
from skgarden import RandomForestQuantileRegressor
from qosa import QuantileRegressionForest
from qosa.tests import ToyInsurance

model = ToyInsurance()

n_samples = 10**4
X = np.asarray(model.input_distribution.getSample(n_samples))
Y = X.sum(axis=1) + np.random.normal(size=n_samples)

# -----------------------------------------------------------------------------
# Compute the conditional quantile estimates 
# -----------------------------------------------------------------------------

#min_leaf = int(np.sqrt(n_samples)*(np.log(n_samples))**1.5)
min_leaf = 5
min_split = min_leaf*2
n_trees = 150
max_features = "auto"
max_samples = None

# With my method
qrf = QuantileRegressionForest(n_estimators=n_trees,
                               max_features=max_features,
                               min_samples_split=min_split, 
                               min_samples_leaf=min_leaf,
                               max_samples=max_samples,
                               n_jobs=-1,
                               random_state=888)
qrf.fit(X=X,y=Y)

alpha = np.asarray([0.2, 0.5, 0.7, 0.9])
n_X_value = 50
X_value = np.asarray(model.input_distribution.getSample(n_X_value))
conditional_quantile_estimate_kevin = qrf.predict(X=X_value, alpha=alpha)

# With Sickit Garden
rfqr = RandomForestQuantileRegressor(n_estimators=n_trees,
                                     max_features=max_features,
                                     min_samples_split=min_split,
                                     min_samples_leaf=min_leaf,
                                     n_jobs=-1,
                                     random_state=888)
rfqr.fit(X=X,y=Y)
conditional_quantile_estimate_skgarden = np.empty((n_X_value, alpha.shape[0]))
for i, alpha_temp in enumerate(alpha):
    conditional_quantile_estimate_skgarden[:, i] = rfqr.predict(X=X_value, quantile=alpha_temp*100)

# -----------------------------------------------------------------------------
# Compute the True value for the conditional quantile
# -----------------------------------------------------------------------------

quantile_gaussian = ss.norm.ppf(q=alpha)
conditional_quantile_true = X_value.sum(axis=1).reshape(-1,1) + quantile_gaussian

# -----------------------------------------------------------------------------
# Compute the RMSE
# -----------------------------------------------------------------------------

RMSE_w_sqrt = (((conditional_quantile_estimate_kevin - conditional_quantile_true)/conditional_quantile_true)**2).mean(axis=0)
RMSE_sqrt = np.sqrt((((conditional_quantile_estimate_kevin - conditional_quantile_true)/conditional_quantile_true)**2).mean(axis=0))

# -----------------------------------------------------------------------------
# Plot 
# -----------------------------------------------------------------------------

x_axis = np.arange(n_X_value)
colors = sns.color_palette('bright')

fig, axes = plt.subplots(figsize=(14,10), nrows=2, ncols= 2)

k = 0
for i in range(2):
    for j in range(2):
        axes[i,j].plot(x_axis, conditional_quantile_estimate_kevin[:,k], linestyle='None', marker='o', markersize=6, color=colors[0], label='Estimate Kev')
        axes[i,j].plot(x_axis, conditional_quantile_estimate_skgarden[:,k], linestyle='None', marker='o', markersize=6, color=colors[1], label='Estimate SK')
        axes[i,j].plot(x_axis, conditional_quantile_true[:,k], linestyle='None', marker='o', markersize=6, color=colors[2], label='True')
        axes[i,j].set_title(r'$\alpha$ = %0.2f   RMSE_without_sqrt = %0.5f'
                             '   RMSE_with_sqrt = %0.5f'% (alpha[k], RMSE_w_sqrt[k], RMSE_sqrt[k]))
        axes[i,j].legend()
        k +=1
fig.suptitle('Minimum number of samples in each leaf node = %d' % (min_leaf, ), y = 1.01)
fig.tight_layout()
fig.savefig('conditional_quantile_with_skgarden_n_samples_%d_n_trees_%d_min_leaf_%d.pdf' %(n_samples, n_trees, min_leaf),
            bbox_inches='tight')