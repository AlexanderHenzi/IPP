# -*- coding: utf-8 -*-


import numpy as np 
from sklearn.ensemble import RandomForestRegressor


def _compute_weight(samples_nodes, X_nodes_k):
    """
    Function to compute the averaged weight of each individual of the original
    training sample in the forest.
    """
    
    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_elements = samples_nodes.size
    
    col_cnt = np.zeros((n_trees), dtype=np.uint32)
    row_idx = np.empty((n_samples + 1), dtype=np.uint32)
    col_idx = np.empty((n_elements), dtype=np.uint32)

    row_idx[0] = 0
    for i in range(n_samples):
        row_idx[i+1] = row_idx[i]
        for j in range(n_trees):
            if samples_nodes[i,j] == X_nodes_k[j]:
                col_cnt[j] += 1
                col_idx[row_idx[i+1]] = j
                row_idx[i+1] += 1

    col_weight = np.empty((n_trees), dtype=np.float64)
    for j in range(n_trees):
        col_weight[j] = 1. / col_cnt[j]

    weighted_mean = np.empty((n_samples), dtype=np.float64)
    for i in range(n_samples):
        s = 0.
        for jj in range(row_idx[i], row_idx[i+1]):
            s += col_weight[col_idx[jj]]
        weighted_mean[i] = s / n_trees

    return weighted_mean


# Sample to calibrate the forest
n_samples = 10**4
X = np.random.exponential(size = (n_samples, 2))
X[:,1] = -X[:,1]
Y = X.sum(axis=1)

# Calibration of the forest with at least 20 observations by leaf
qrf = RandomForestRegressor(n_estimators=10**2, min_samples_split=40, min_samples_leaf=20, n_jobs=-1)
qrf.fit(X, Y)

# Get the leaf's number of each individual for each tree within the sample training
samples_nodes = qrf.apply(X)

# Compute the weights related to the sample training for these 5 new observations
X_nodes_values = np.random.exponential(size = (5, 2))    
X_nodes_values[:,1] = -X_nodes_values[:,1]
X_nodes = qrf.apply(X_nodes_values)

weight_0 = _compute_weight(samples_nodes, X_nodes[0,:])
weight_1 = _compute_weight(samples_nodes, X_nodes[1,:])
weight_2 = _compute_weight(samples_nodes, X_nodes[2,:])
