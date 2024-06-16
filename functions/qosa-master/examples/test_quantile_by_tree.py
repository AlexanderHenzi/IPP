# -*- coding: utf-8 -*-

import time
import numba as nb
import numpy as np
from qosa import QuantileRegressionForest
from warnings import warn


@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:])", nogil=True, cache=False, parallel=False)
def _compute_conditional_quantile_by_tree_bootstrap_1(output_samples, samples_nodes, idx_bootstrap_samples, X_nodes, alpha):
    """
    Add docstring
    """
    
    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_quantiles = X_nodes.shape[0]
    n_alphas = alpha.shape[0]
    
    X_nodes = np.asfortranarray(X_nodes)
    
    # Unique leaf nodes of each tree
    samples_nodes = np.asfortranarray(samples_nodes)
    idx_leaves_by_tree = [np.unique(samples_nodes[:, i]) for i in range(n_trees)]
    
    quantiles = np.zeros((n_quantiles, n_alphas), dtype=np.float64)
    Y_leaves = np.empty((n_samples), dtype=np.float64)
    idx_X_nodes_in_current_leaf = np.empty((n_quantiles), dtype=np.int64)
    
    for i in range(n_trees):
        idx_leaves_for_tree_i = idx_leaves_by_tree[i]
        
        for idx_current_leaf in idx_leaves_for_tree_i:
            n_idx_X_nodes_in_current_leaf = 0
            for j in range(n_quantiles):
                if X_nodes[j, i] == idx_current_leaf:
                    idx_X_nodes_in_current_leaf[n_idx_X_nodes_in_current_leaf] = j
                    n_idx_X_nodes_in_current_leaf += 1
            
            n_Y_leaves = 0    
            for k in range(n_samples):
                if samples_nodes[idx_bootstrap_samples[k, i], i] == idx_current_leaf:
                    Y_leaves[n_Y_leaves] = output_samples[idx_bootstrap_samples[k, i]]
                    n_Y_leaves += 1
                                
            quantiles_in_leaf = np.percentile(Y_leaves[:n_Y_leaves], alpha*100)
            for l in range(n_idx_X_nodes_in_current_leaf):
                quantiles[idx_X_nodes_in_current_leaf[l], :] += quantiles_in_leaf
                
    return quantiles/n_trees


@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:], boolean)", nogil=True, cache=False, parallel=True)
def _compute_conditional_quantile_by_tree_bootstrap_2(output_samples, samples_nodes, idx_bootstrap_samples, X_nodes, alpha, used_bootstrap_samples):
    """
    Add docstring
    """
    
    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_quantiles = X_nodes.shape[0]
    n_alphas = alpha.shape[0]
    
    X_nodes = np.asfortranarray(X_nodes)

    # Unique leaf nodes of each tree
    samples_nodes = np.asfortranarray(samples_nodes)
    idx_leaves_by_tree = [np.unique(samples_nodes[:, i]) for i in range(n_trees)]
    
    quantiles = np.empty((n_trees, n_quantiles, n_alphas), dtype=np.float64)
    
    for i in nb.prange(n_trees):
        idx_leaves_for_tree_i = idx_leaves_by_tree[i]
        Y_leaves = np.empty((n_samples), dtype=np.float64)
        idx_X_nodes_in_current_leaf = np.empty((n_quantiles), dtype=np.int64)
        
        for idx_current_leaf in idx_leaves_for_tree_i:            
            n_idx_X_nodes_in_current_leaf = 0
            for j in range(n_quantiles):
                if X_nodes[j, i] == idx_current_leaf:
                    idx_X_nodes_in_current_leaf[n_idx_X_nodes_in_current_leaf] = j
                    n_idx_X_nodes_in_current_leaf += 1
            
            n_Y_leaves= 0
            if used_bootstrap_samples:
                for k in range(n_samples):
                    if samples_nodes[idx_bootstrap_samples[k, i], i] == idx_current_leaf:
                        Y_leaves[n_Y_leaves] = output_samples[idx_bootstrap_samples[k, i]]
                        n_Y_leaves += 1
            else:
                for k in range(n_samples):
                    if samples_nodes[k, i] == idx_current_leaf:
                        Y_leaves[n_Y_leaves] = output_samples[k]
                        n_Y_leaves += 1
                        
            quantiles_in_leaf = np.percentile(Y_leaves[:np.int64(n_Y_leaves)], alpha*100)            
            for l in range(n_idx_X_nodes_in_current_leaf):
                quantiles[i, idx_X_nodes_in_current_leaf[l], :] = quantiles_in_leaf
                
    return quantiles.sum(axis=0)/n_trees


@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:])", nogil=True, cache=False, parallel=False)
def _compute_conditional_quantile_by_tree_bootstrap_3(output_samples, samples_nodes, idx_bootstrap_samples, X_nodes, alpha):
    """
    Add docstring
    """
    
    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_quantiles = X_nodes.shape[0]
    n_alphas = alpha.shape[0]
    
    X_nodes = np.asfortranarray(X_nodes)
    samples_nodes = np.asfortranarray(samples_nodes)
    
    quantiles = np.zeros((n_quantiles, n_alphas), dtype=np.float64)
    Y_leaves = np.empty((n_samples), dtype=np.float64)
    
    for i in range(n_trees):
        X_nodes_by_column = np.unique(X_nodes[:, i])
        
        for X_node in X_nodes_by_column:    
            n_Y_leaves = 0    
            for j in range(n_samples):
                if samples_nodes[idx_bootstrap_samples[j, i], i] == X_node:
                    Y_leaves[n_Y_leaves] = output_samples[idx_bootstrap_samples[j, i]]
                    n_Y_leaves += 1
            
            quantiles_in_leaf = np.percentile(Y_leaves[:n_Y_leaves], alpha*100)

            idx_inverse_X_node = np.where(X_nodes[:, i] == X_node)[0]
            for k in idx_inverse_X_node:
                quantiles[k, :] += quantiles_in_leaf

    return quantiles/n_trees


# If I want to use parallel=True, I have to remove the condition "if" with used_bootstrap_samples
@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:], boolean)", nogil=True, cache=False, parallel=False)
def _compute_conditional_quantile_by_tree_bootstrap_4(output_samples, samples_nodes, idx_bootstrap_samples, X_nodes, alpha, used_bootstrap_samples):
    """
    Add docstring
    """
    
    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_quantiles = X_nodes.shape[0]
    n_alphas = alpha.shape[0]
    
    X_nodes = np.asfortranarray(X_nodes)
    samples_nodes = np.asfortranarray(samples_nodes)
    
    quantiles = np.empty((n_trees, n_quantiles, n_alphas), dtype=np.float64)
    
    for i in range(n_trees):
        X_nodes_by_column = np.unique(X_nodes[:, i])
        
        for j in nb.prange(X_nodes_by_column.shape[0]):
            X_node = X_nodes_by_column[j]
            Y_leaves = np.empty((n_samples), dtype=np.float64)
            
            n_Y_leaves = 0
            if used_bootstrap_samples:
                for k in range(n_samples):    
                    if samples_nodes[idx_bootstrap_samples[k, i], i] == X_node:
                        Y_leaves[n_Y_leaves] = output_samples[idx_bootstrap_samples[k, i]]
                        n_Y_leaves += 1
            else:
                for k in range(n_samples):
                    if samples_nodes[k, i] == X_node:
                        Y_leaves[n_Y_leaves] = output_samples[k]
                        n_Y_leaves += 1
                              
            quantiles_in_leaf = np.percentile(Y_leaves[:n_Y_leaves], alpha*100)
            for l in range(n_quantiles):
                if X_nodes[l, i] == X_node:
                    quantiles[i, l, :] = quantiles_in_leaf

    return quantiles.sum(axis=0)/n_trees

# -----------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

qrf= QuantileRegressionForest(n_estimators=n_trees, min_samples_split=n_min_split, min_samples_leaf=n_min_leaf, n_jobs=-1)
qrf.fit(X1, Y)

samples_nodes = qrf.apply(X1)
output_samples = Y
X_nodes_val = np.random.exponential(size = (n_samples, 1))
X_nodes = qrf.apply(X_nodes_val)
alpha = np.array([0.3, 0.5, 0.9])

idx_bootstrap_samples, inbag_samples = qrf.compute_inbag_samples()

start = time.time()
_, ind, rev_ind = np.unique(X_nodes, axis=0, return_index=True, return_inverse=True)
for i in range(1):
   a =  _compute_conditional_quantile_by_tree_bootstrap_1(output_samples, samples_nodes, idx_bootstrap_samples, X_nodes[ind], alpha)
   a = a[rev_ind]
print('Time for the first : ', time.time() - start)

start = time.time()
_, ind, rev_ind = np.unique(X_nodes, axis=0, return_index=True, return_inverse=True)
for i in range(10):
   b =  _compute_conditional_quantile_by_tree_bootstrap_2(output_samples, samples_nodes, idx_bootstrap_samples, X_nodes[ind], alpha, False)
   b = b[rev_ind]
print('Time for the second : ', time.time() - start)

start = time.time()
_, ind, rev_ind = np.unique(X_nodes, axis=0, return_index=True, return_inverse=True)
for i in range(1):
   c =  _compute_conditional_quantile_by_tree_bootstrap_3(output_samples, samples_nodes, idx_bootstrap_samples, X_nodes[ind], alpha)
   c = c[rev_ind]
print('Time for the third : ', time.time() - start)

start = time.time()
_, ind, rev_ind = np.unique(X_nodes, axis=0, return_index=True, return_inverse=True)
for i in range(10):
   d =  _compute_conditional_quantile_by_tree_bootstrap_4(output_samples, samples_nodes, idx_bootstrap_samples, X_nodes[ind], alpha, True)
   d = d[rev_ind]
print('Time for the fourth : ', time.time() - start)

print((a != b).sum(axis=0))
print((b != c).sum(axis=0))
print((c != d).sum(axis=0))
print((b != d).sum(axis=0))

#start = time.time()
p = qrf.predict(X_nodes_val, alpha, 'toto', used_bootstrap_samples=False)
#print(time.time() - start)




# -----------------------------------------------------------------------------
#
# Implementation of the OOB error
#
# -----------------------------------------------------------------------------

n_samples = 10**3
n_trees = 10**1

# Size of the leaf nodes
n_min_leaf = 20
n_min_split = n_min_leaf*2

# Use one sample to calibrate the forest, the quantiles and the indices
X = np.random.exponential(size = (n_samples, 2))
Y = X[:,0] - X[:,1]
X1 = X[:,0].reshape(-1,1)
alpha = np.array([0.3, 0.5, 0.9])

qrf= QuantileRegressionForest(n_estimators=n_trees, 
                              min_samples_split=n_min_split, 
                              min_samples_leaf=n_min_leaf, 
                              n_jobs=-1)
qrf.fit(X1, Y, oob_score_quantile=True, alpha=alpha, used_bootstrap_samples=False)

samples_nodes = qrf.apply(X1)
output_samples = Y

idx_bootstrap_samples, _ = qrf._compute_inbag_samples()


@nb.njit("float64[:](float64[:,:], float64[:])", nogil=True, cache=False, parallel=True)
def _averaged_check_function_alpha_array(u, alpha):
    """
    Definition of the check function also called pinball loss function.
    """
    n_alphas = alpha.shape[0]
    n_samples = u.shape[0]

    check_function = np.empty((n_alphas, n_samples), dtype=np.float64)
    for i in nb.prange(n_samples):
        for j in range(n_alphas):
            check_function[j,i] = u[i,j]*(alpha[j] - (u[i,j] < 0.))

    averaged_check_function = np.empty((n_alphas), dtype=np.float64)
    for i in nb.prange(n_alphas):
        averaged_check_function[i] = check_function[i,:].mean()

    return averaged_check_function

from sklearn.ensemble.forest import _generate_unsampled_indices

def compute_oob_samples(forest):
    idx_oob_samples = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.int32[:])
    n_trees = forest.n_estimators
    n_samples = forest._n_samples  
    
    # Here at each iteration we obtain out of bag samples for every tree.
    for i in range(n_trees):
        idx_oob_samples.update({i : _generate_unsampled_indices(forest.estimators_[i].random_state, 
                                                                n_samples,
                                                                n_samples)})
        
    return idx_oob_samples

@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], DictType(int64, int32[:]), float64[:])", nogil=True, cache=False, parallel=True)
def _compute_conditional_quantile_for_oob_samples_on_each_tree( 
                                                       output_samples,
                                                       samples_nodes,
                                                       idx_bootstrap_samples,
                                                       idx_oob_samples,
                                                       alpha):
    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_alphas = alpha.shape[0]

    # Unique leaf nodes of each tree
    samples_nodes = np.asfortranarray(samples_nodes)
    idx_leaves_by_tree = [np.unique(samples_nodes[:, i]) for i in range(n_trees)]
    
    quantiles = np.zeros((n_trees, n_samples, n_alphas), dtype=np.float64)
    n_quantiles = np.zeros((n_trees, n_samples), dtype=np.uint32)
    
    for i in nb.prange(n_trees):
        idx_leaves_for_tree_i = idx_leaves_by_tree[i]
        X_nodes_oob_for_tree_i = samples_nodes[:, i][idx_oob_samples[np.int64(i)]]
        n_X_nodes_oob_for_tree_i = X_nodes_oob_for_tree_i.shape[0]
        Y_leaves = np.empty((n_samples), dtype=np.float64)
        idx_X_nodes_oob_in_current_leaf = np.empty((n_X_nodes_oob_for_tree_i), dtype=np.uint32)
        
        for idx_current_leaf in idx_leaves_for_tree_i:            
            n_idx_X_nodes_oob_in_current_leaf = 0
            for j in range(n_X_nodes_oob_for_tree_i):
                if X_nodes_oob_for_tree_i[j] == idx_current_leaf:
                    idx_X_nodes_oob_in_current_leaf[n_idx_X_nodes_oob_in_current_leaf] = j
                    n_idx_X_nodes_oob_in_current_leaf += 1
            
            n_Y_leaves = 0
            for k in range(n_samples):
                if samples_nodes[idx_bootstrap_samples[k, i], i] == idx_current_leaf:
                    Y_leaves[n_Y_leaves] = output_samples[idx_bootstrap_samples[k, i]]
                    n_Y_leaves += 1
                        
            quantiles_in_leaf = np.percentile(Y_leaves[:np.int64(n_Y_leaves)], alpha*100)            
            for l in range(n_idx_X_nodes_oob_in_current_leaf):
                quantiles[i, idx_oob_samples[np.int64(i)][idx_X_nodes_oob_in_current_leaf[l]], :quantiles.shape[2]] = quantiles_in_leaf
                n_quantiles[i, idx_oob_samples[np.int64(i)][idx_X_nodes_oob_in_current_leaf[l]]] = 1
    
    n_quantiles = n_quantiles.sum(axis=0)
    n_quantiles[n_quantiles == 0] = 1
    
    return quantiles.sum(axis=0)/n_quantiles.reshape(-1,1)

idx_oob_samples = compute_oob_samples(qrf)

def _set_oob_score(self, X, y):
   """
   Compute out-of-bag scores.
   """

   X = self._input_samples
   y = self._output_samples

   n_samples = self._n_samples
   
   idx_oob_samples = compute_oob_samples(self)
   
   conditional_quantiles =_compute_conditional_quantile_for_oob_samples_on_each_tree( 
                                                           output_samples,
                                                           samples_nodes,
                                                           idx_bootstrap_samples,
                                                           idx_oob_samples,
                                                           alpha)

   if (conditional_quantiles.sum(axis=1) == 0).any():
       warn("Some inputs do not have OOB scores. "
            "This probably means too few trees were used "
            "to compute any reliable oob estimates.")

   self.oob_prediction_ = conditional_quantiles
   self.oob_score_ = _averaged_check_function_alpha_array(
                                        y.reshape(-1,1)-conditional_quantiles,
                                        alpha)
   
   
   
@nb.njit("float64[:,:](float64[:], int64[:,:], DictType(int64, int32[:]), float64[:])", nogil=True, cache=False, parallel=True)
def _compute_conditional_quantile_for_oob_samples_on_each_tree_with_original_data( 
                                                       output_samples,
                                                       samples_nodes,
                                                       idx_oob_samples,
                                                       alpha):
    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_alphas = alpha.shape[0]

    # Unique leaf nodes of each tree
    samples_nodes = np.asfortranarray(samples_nodes)
    idx_leaves_by_tree = [np.unique(samples_nodes[:, i]) for i in range(n_trees)]
    
    quantiles = np.zeros((n_trees, n_samples, n_alphas), dtype=np.float64)
    n_quantiles = np.zeros((n_trees, n_samples), dtype=np.uint32)
    
    for i in nb.prange(n_trees):
        idx_leaves_for_tree_i = idx_leaves_by_tree[i]
        X_nodes_oob_for_tree_i = samples_nodes[:, i][idx_oob_samples[np.int64(i)]]
        n_X_nodes_oob_for_tree_i = X_nodes_oob_for_tree_i.shape[0]
        idx_Y_in_current_leaf = np.empty((n_samples), dtype=np.uint32)
        idx_X_nodes_oob_in_current_leaf = np.empty((n_X_nodes_oob_for_tree_i), dtype=np.uint32)
        
        for idx_current_leaf in idx_leaves_for_tree_i:            
            n_idx_X_nodes_oob_in_current_leaf = 0
            for j in range(n_X_nodes_oob_for_tree_i):
                if X_nodes_oob_for_tree_i[j] == idx_current_leaf:
                    idx_X_nodes_oob_in_current_leaf[n_idx_X_nodes_oob_in_current_leaf] = j
                    n_idx_X_nodes_oob_in_current_leaf += 1
            
            n_idx_Y_in_current_leaf = 0
            for k in range(n_samples):
                if samples_nodes[k, i] == idx_current_leaf:
                    idx_Y_in_current_leaf[n_idx_Y_in_current_leaf] = k
                    n_idx_Y_in_current_leaf += 1
                        
            for l in range(n_idx_X_nodes_oob_in_current_leaf):
                quantiles_in_leaf = np.percentile(
                    output_samples[
                        idx_Y_in_current_leaf[:np.int64(n_idx_Y_in_current_leaf)][
                            idx_Y_in_current_leaf[:np.int64(n_idx_Y_in_current_leaf)] != 
                            idx_oob_samples[np.int64(i)][idx_X_nodes_oob_in_current_leaf[l]]]
                        ],
                    alpha*100)
                quantiles[i, 
                          idx_oob_samples[np.int64(i)][idx_X_nodes_oob_in_current_leaf[l]],
                          :quantiles.shape[2]] = quantiles_in_leaf
                n_quantiles[i, idx_oob_samples[np.int64(i)][idx_X_nodes_oob_in_current_leaf[l]]] = 1
    
    n_quantiles = n_quantiles.sum(axis=0)
    n_quantiles[n_quantiles == 0] = 1
    
    return quantiles.sum(axis=0)/n_quantiles.reshape(-1,1)

conditional_quantiles = _compute_conditional_quantile_for_oob_samples_on_each_tree_with_original_data( 
                                                       output_samples,
                                                       samples_nodes,
                                                       idx_oob_samples,
                                                       alpha)