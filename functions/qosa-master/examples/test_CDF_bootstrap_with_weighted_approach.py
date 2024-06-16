# -*- coding: utf-8 -*-

import time
import numba as nb
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from qosa import QuantileRegressionForest

# ------------------------
#
# Function for the package
#
# ------------------------

from sklearn.ensemble.forest import _generate_sample_indices

def compute_inbag_samples(n_samples, forest):
    n_trees = forest.n_estimators
    idx_bootstrap_samples = np.empty((n_samples, n_trees), dtype=np.uint32, order='F')
    inbag_samples = np.empty((n_samples, n_trees), dtype=np.uint32, order='F')
    
    for idx_tree in range(n_trees):
            idx_bootstrap_samples[:,idx_tree] = _generate_sample_indices(
                                                    forest.estimators_[idx_tree].random_state,
                                                    n_samples,
                                                    n_samples)
            inbag_samples[:,idx_tree] = np.bincount(idx_bootstrap_samples[:,idx_tree],
                                                    minlength=n_samples)
    return idx_bootstrap_samples, inbag_samples


# -----------------------------------------------------------------------------
# NINTH CODE
# -----------------------------------------------------------------------------
  
@nb.njit("void(float64[:], int64, int64[:,:], uint32[:,:], int64)", nogil=True, cache=False, parallel=False)
def conditional_CDF_7(CDF_by_tree, idx_tree, samples_nodes_sorted, inbag_samples_sorted, X_node):
    n_samples = samples_nodes_sorted.shape[0]
    
    count = 0.
    CDF_by_tree[:] = 0.
    for i in range(n_samples):
        if inbag_samples_sorted[i, idx_tree] != 0:
            if samples_nodes_sorted[i, idx_tree] == X_node:
                count += inbag_samples_sorted[i, idx_tree]
                CDF_by_tree[i:] += inbag_samples_sorted[i, idx_tree]

    CDF_by_tree /= count

@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:])", nogil=True, cache=False, parallel=True)
def _compute_conditional_quantile_with_bootstrap_data_10_bis(output_samples,
                                                         samples_nodes,
                                                         inbag_samples,
                                                         X_nodes,
                                                         alpha):
    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_quantiles = X_nodes.shape[0]
    n_alphas = alpha.shape[0]
    quantiles = np.empty((n_quantiles, n_alphas), dtype=np.float64)
    
    X_nodes = np.asfortranarray(X_nodes)
    
    order_statistic = np.argsort(output_samples)
    samples_nodes_sorted = np.asfortranarray(samples_nodes[order_statistic])
    inbag_samples_sorted = np.asfortranarray(inbag_samples[order_statistic])

    CDF_forest = np.empty((n_quantiles, n_samples), dtype=np.float64)
    
    for j in range(n_trees):
        X_nodes_by_column = np.unique(X_nodes[:, j])
        for i in nb.prange(X_nodes_by_column.shape[0]):
            CDF_by_tree = np.empty((n_samples), dtype=np.float64)
            X_node = X_nodes_by_column[i]
            conditional_CDF_7(CDF_by_tree,
                              j,
                              samples_nodes_sorted,
                              inbag_samples_sorted,
                              X_node)
            
            for k in range(n_quantiles):
                if X_nodes[k, j] == X_node:
                    CDF_forest[k, :] += CDF_by_tree

    for i in nb.prange(n_quantiles):
        CDF_forest[i,:] /= n_trees
        quantiles[i,:] = np.array([
                        output_samples[
                            order_statistic[
                                np.argmax((CDF_forest[i,:] >= alpha_var).astype(np.uint32))]
                                    ] 
                        for alpha_var in alpha])

    return quantiles


# -----------------------------------------------------------------------------
# TENTH CODE
# -----------------------------------------------------------------------------
  
@nb.njit("float64[:](int64[:,:], uint32[:,:], int64[:])", nogil=True, cache=False, parallel=False)
def _compute_weight_with_bootstrap_data(samples_nodes, inbag_samples, X_nodes_k):
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
                col_cnt[j] += inbag_samples[i,j]
                col_idx[row_idx[i+1]] = j
                row_idx[i+1] += 1

    col_weight = np.empty((n_trees), dtype=np.float64)
    for j in range(n_trees):
        col_weight[j] = 1. / col_cnt[j]

    weighted_mean = np.empty((n_samples), dtype=np.float64)
    for i in range(n_samples):
        s = 0.
        for jj in range(row_idx[i], row_idx[i+1]):
            s += inbag_samples[i, col_idx[jj]] * col_weight[col_idx[jj]]
        weighted_mean[i] = s / n_trees

    return weighted_mean


# --------------------------------------
# For the class QuantileRegressionForest
# --------------------------------------

@nb.njit("void(float64[:,:], float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:])", nogil=True, cache=False, parallel=True)
def _compute_conditional_quantile_with_bootstrap_data(quantiles,
                                                      output_samples,
                                                      samples_nodes,
                                                      inbag_samples,
                                                      X_nodes,
                                                      alpha):
    """
    "Weighted_CDF" : Function to compute the conditional quantiles thanks to the weights 
    """
    
    n_quantiles = quantiles.shape[0]
    order_statistic = np.argsort(output_samples)

    # For each observation to compute
    for i in nb.prange(n_quantiles):     
        # Compute the conditional weights associated to each individual
        weight = _compute_weight_with_bootstrap_data(samples_nodes, 
                                                     inbag_samples, 
                                                     X_nodes[i, :])
        
        # Compute the quantiles thanks to the Cumulative Distribution Function 
        # for each value of alpha
        CDF = np.cumsum(weight[order_statistic])
   
        quantiles[i, :] = np.array([
                            output_samples[
                                order_statistic[
                                    np.argmax((CDF >= alpha_var).astype(np.uint32))]
                                          ] 
                            for alpha_var in alpha])


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

n_samples = 10**4
n_trees = 10**2

# Size of the leaf nodes
n_min_leaf = 20
n_min_split = n_min_leaf*2

# Use one sample to calibrate the forest, the quantiles and the indices
X = np.random.exponential(size = (n_samples, 2))
Y = X[:,0] - X[:,1]
X1 = X[:,0].reshape(-1,1)

quantForest = QuantileRegressionForest(n_estimators=n_trees, min_samples_split=n_min_split, min_samples_leaf=n_min_leaf, n_jobs=-1)
quantForest.fit(X1, Y)

samples_nodes = quantForest._samples_nodes
output_samples = Y
X_nodes_val = np.random.exponential(size = (n_samples, 1))    
X_nodes = quantForest.get_nodes(X_nodes_val)
alpha = np.array([0.3, 0.5, 0.9])
n_alphas = alpha.shape[0]

idx_bootstrap_samples, inbag_samples = compute_inbag_samples(n_samples, quantForest)

start = time.time()
_, ind, rev_ind = np.unique(X_nodes, axis=0, return_index=True, return_inverse=True)
for i in range(1):
   a = _compute_conditional_quantile_with_bootstrap_data_10_bis(output_samples, samples_nodes, inbag_samples, X_nodes[ind], alpha)
   a = a[rev_ind]
print('Time for the ninth bis: ', time.time() - start)

start = time.time()
_, ind, rev_ind = np.unique(X_nodes, axis=0, return_index=True, return_inverse=True)
for i in range(1):
    b = np.empty((ind.shape[0], n_alphas), dtype=np.float64, order='C')
    _compute_conditional_quantile_with_bootstrap_data(b, output_samples, samples_nodes, inbag_samples, X_nodes[ind], alpha)
    b = b[rev_ind]
print('Time for the tenth: ', time.time() - start)

start = time.time()
f = quantForest.predict(X_nodes_val, alpha)
print(time.time() - start)

print((a != b).sum(axis=0))
print((a != f).sum(axis=0))
print((b != f).sum(axis=0))