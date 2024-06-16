# -*- coding: utf-8 -*-

import numpy as np
from qosa import QuantileRegressionForest


np.random.seed(0)

dim = 2
alpha = np.array([0.5, 0.7, 0.99])
p = 1./(1-alpha)

n_sample = 10**4
n_tree = 2*10**2
n_RMSE = 10**0

# Size of the leaf nodes
n_min_leaf = np.unique(np.logspace(np.log10(5), np.log10(1500), 10, endpoint=True, dtype=int))
n_min_split = n_min_leaf*2

QOSA_indices = np.zeros((alpha.shape[0], n_min_leaf.shape[0], n_RMSE)) # Only for the first variable at the moment
QOSA_indices_bis = np.zeros_like(QOSA_indices) # Only for the first variable at the moment

for k in range(n_RMSE):
    print(k)
    
    # Use one sample to calibrate the forest, the quantiles and the indices
    X = np.random.exponential(size = (n_sample,2))
    Y = X[:,0] - X[:,1]
    X1 = X[:,0].reshape(-1,1)
    
    meanY = Y.mean()
    
    # Use a second sample to calibrate the forest and compute the quantiles
    X_bis = np.random.exponential(size = (n_sample,2))
    Y_bis = X_bis[:,0] - X_bis[:,1]
    X1_bis = X_bis[:,0].reshape(-1,1)
    
    theta_bis = np.percentile(Y_bis, q=alpha*100)
    denominateur_bis = [p_temp*(Y*(Y>theta_temp)).mean() - meanY for p_temp, theta_temp in zip(p,theta_bis)]
    
    for j,n_min_leaf_temp in enumerate(n_min_leaf):
        n_min_split_temp = n_min_split[j]
               
        quantForest = QuantileRegressionForest(n_estimators=n_tree, min_samples_split=n_min_split_temp, min_samples_leaf=n_min_leaf_temp, n_jobs=-1)
        quantForest.fit(X1_bis, Y_bis)
        conditional_quantiles = quantForest.predict(X1, alpha)
        numerator = (Y[:,np.newaxis]*(Y[:,np.newaxis]>conditional_quantiles)).mean(axis=0)*p - meanY
        print(numerator*(1-alpha))