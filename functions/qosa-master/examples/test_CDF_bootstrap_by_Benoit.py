# -*- coding: utf-8 -*-

import time
import numba as nb
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import psutil

NUMBERS_OF_TREES = 3*10**2

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

def node_count_forest(forest):
    n_trees = forest.n_estimators
    node_count_by_tree = np.zeros((n_trees), dtype=np.uint32)
    
    for i in range(n_trees):
        node_count_by_tree[i] = forest.estimators_[i].tree_.__getstate__()['node_count']
        
    return np.amax(node_count_by_tree)

def node_final_count_forest(forest):
    n_trees = forest.n_estimators
    node_final_count_by_tree = np.zeros((n_trees), dtype=np.uint32)
    
    for i in range(n_trees):
        tab = forest.estimators_[i].tree_.__getstate__()['nodes']
        node_final_count_by_tree[i] = np.argwhere(tab['left_child'] == tab['right_child']).shape[0]
        
    return node_final_count_by_tree, np.amax(node_final_count_by_tree)

# -----------------------------------------------------------------------------
# THIRD CODE
# -----------------------------------------------------------------------------

@nb.njit("float64[:](float64[:,:,:], int64[:,:], uint32[:,:], int64[:])", nogil=True, cache=False, parallel=True)
def conditional_CDF_3(cache, samples_nodes_sorted, inbag_samples_sorted, X_nodes):
    n_samples = samples_nodes_sorted.shape[0]
    n_trees = samples_nodes_sorted.shape[1]
    
    CDF_by_tree = np.zeros((n_samples, n_trees), dtype=np.float64)
    CDF_forest = np.empty((n_samples), dtype=np.float64)
    
    for i in nb.prange(n_trees):
        if cache[X_nodes[i],i,0] < 0:
            count = 0.
            for j in range(n_samples):
                if inbag_samples_sorted[j,i] != 0:
                    if samples_nodes_sorted[j,i] == X_nodes[i]:
                        count += inbag_samples_sorted[j,i]
                        for k in range(j, n_samples):
                            CDF_by_tree[k,i] += inbag_samples_sorted[j,i]
            
            for l in range(n_samples):
                CDF_by_tree[l,i] /= count
                cache[X_nodes[i],i,l] = CDF_by_tree[l,i]
        else:
            for l in range(n_samples):
                CDF_by_tree[l,i] = cache[X_nodes[i],i,l]
    
    for i in nb.prange(n_samples):
        res = 0.
        for j in range(n_trees):
            res += CDF_by_tree[i,j]
        CDF_forest[i] = res/n_trees

    return CDF_forest


@nb.njit("float64[:,:](int64, float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:])",nogil=True, cache=False, parallel=False)
def _compute_conditional_quantile_with_bootstrap_data_4(node_count_max, output_samples, samples_nodes, inbag_samples, X_nodes, alpha):
    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_quantiles = X_nodes.shape[0]
    n_alphas = alpha.shape[0]
    quantiles = np.zeros((n_quantiles, n_alphas), dtype=np.float64)
    
    order_statistic = np.argsort(output_samples)
    samples_nodes_sorted = samples_nodes[order_statistic]
    inbag_samples_sorted = inbag_samples[order_statistic]

    cache = -np.ones((node_count_max, n_trees, n_samples), dtype=np.float64)
    
    for i in range(n_quantiles):
        CDF_forest = conditional_CDF_3(cache, 
                                       samples_nodes_sorted,
                                       inbag_samples_sorted,
                                       X_nodes[i,:])
        
        quantiles[i, :] = np.array([
                            output_samples[
                                order_statistic[
                                    np.argmax((CDF_forest >= alpha_var).astype(np.uint32))]
                                        ] 
                            for alpha_var in alpha])

    return quantiles

# -----------------------------------------------------------------------------
# FOURTH CODE
# -----------------------------------------------------------------------------

@nb.njit("float64[:](float64[:,:], DictType(UniTuple(int64, 2), float64[:]),int64[:,:], uint32[:,:], int64[:])", nogil=True, cache=False, parallel=False)
def conditional_CDF_4(CDF_by_tree, cache, samples_nodes_sorted, inbag_samples_sorted, X_nodes):
    n_samples = samples_nodes_sorted.shape[0]
    n_trees = samples_nodes_sorted.shape[1]
    
    for i in range(n_trees):
        key_tree = (X_nodes[i], i)
        if not bool(key_tree in cache):
            count = 0.
            CDF_by_tree[i, :] = 0.
            for j in range(n_samples):
                if inbag_samples_sorted[j, i] != 0:
                    if samples_nodes_sorted[j, i] == X_nodes[i]:
                        count += inbag_samples_sorted[j, i]
                        CDF_by_tree[i, j:] += inbag_samples_sorted[j, i]

            CDF_by_tree[i, :] /= count
            cache[key_tree] = CDF_by_tree[i, :].copy()
        else:
            CDF_by_tree[i, :] = cache[key_tree]

    return CDF_by_tree.sum(axis=0) / n_trees


dict_key_type = nb.types.UniTuple(nb.types.int64, 2)
dict_value_type = nb.types.float64[:]
@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:], int64)", nogil=True, cache=False, parallel=False)
def _compute_conditional_quantile_with_bootstrap_data_5(output_samples, samples_nodes, inbag_samples, X_nodes, alpha, max_cache_length):
    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_quantiles = X_nodes.shape[0]
    n_alphas = alpha.shape[0]
    quantiles = np.zeros((n_quantiles, n_alphas), dtype=np.float64)
    
    order_statistic = np.argsort(output_samples)
    samples_nodes_sorted = samples_nodes[order_statistic]
    inbag_samples_sorted = inbag_samples[order_statistic]

    tree_cache = nb.typed.Dict.empty(key_type=dict_key_type, value_type=dict_value_type)
    CDF_by_tree = np.empty((n_trees, n_samples), dtype=np.float64)

    for i in range(n_quantiles):
#        print(len(tree_cache), "/", max_cache_length)
        if len(tree_cache) > max_cache_length:
            for j in range(len(tree_cache) - max_cache_length):
                tree_cache.popitem()

        CDF_forest = conditional_CDF_4(CDF_by_tree,
                                       tree_cache, 
                                       samples_nodes_sorted,
                                       inbag_samples_sorted,
                                       X_nodes[i,:])

        quantiles[i, :] = np.array([
                            output_samples[
                                order_statistic[
                                    np.argmax((CDF_forest >= alpha_var).astype(np.uint32))]
                                        ] 
                            for alpha_var in alpha])

    return quantiles


# -----------------------------------------------------------------------------
# FIFTH CODE
# -----------------------------------------------------------------------------

from numba.unsafe.ndarray import to_fixed_tuple
dict_tree_key_type = nb.types.UniTuple(nb.types.int64, 2)
dict_tree_value_type = nb.types.float64[:]
dict_forest_key_type = nb.types.UniTuple(nb.types.int64, NUMBERS_OF_TREES)
dict_forest_value_type = nb.types.float64[:]
@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:], int64)", nogil=True, cache=False, parallel=False)
def _compute_conditional_quantile_with_bootstrap_data_6(output_samples, samples_nodes, inbag_samples, X_nodes, alpha, max_cache_length):
    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_quantiles = X_nodes.shape[0]
    n_alphas = alpha.shape[0]
    quantiles = np.zeros((n_quantiles, n_alphas), dtype=np.float64)
    
    order_statistic = np.argsort(output_samples)
    samples_nodes_sorted = np.asfortranarray(samples_nodes[order_statistic])
    inbag_samples_sorted = np.asfortranarray(inbag_samples[order_statistic])

    tree_cache = nb.typed.Dict.empty(key_type=dict_tree_key_type, value_type=dict_tree_value_type)
    forest_cache = nb.typed.Dict.empty(key_type=dict_forest_key_type, value_type=dict_forest_value_type)
    CDF_by_tree = np.empty((n_trees, n_samples), dtype=np.float64)

    for i in range(n_quantiles):
        key_forest = to_fixed_tuple(X_nodes[i,:], NUMBERS_OF_TREES)
        if not bool(key_forest in forest_cache):
            if len(tree_cache) > max_cache_length:
                for j in range(len(tree_cache) - max_cache_length):
                    tree_cache.popitem()
    
            CDF_forest = conditional_CDF_4(CDF_by_tree,
                                           tree_cache, 
                                           samples_nodes_sorted,
                                           inbag_samples_sorted,
                                           X_nodes[i,:])
    
            quantiles[i, :] = np.array([
                                output_samples[
                                    order_statistic[
                                        np.argmax((CDF_forest >= alpha_var).astype(np.uint32))]
                                            ] 
                                for alpha_var in alpha])
            forest_cache[key_forest] = quantiles[i, :].copy()
        else:
            quantiles[i, :] = forest_cache[key_forest]
    return quantiles


# -----------------------------------------------------------------------------
# SIXTH CODE
# -----------------------------------------------------------------------------

@nb.njit("float64[:](float64[:,:],int64[:,:], uint32[:,:], int64[:])", nogil=True, cache=False, parallel=True)
def conditional_CDF_5(CDF_by_tree, samples_nodes_sorted, inbag_samples_sorted, X_nodes):
    n_samples = samples_nodes_sorted.shape[0]
    n_trees = samples_nodes_sorted.shape[1]
    
    for i in nb.prange(n_trees):
        count = 0.
        CDF_by_tree[i, :] = 0.
        for j in range(n_samples):
            if inbag_samples_sorted[j, i] != 0:
                if samples_nodes_sorted[j, i] == X_nodes[i]:
                    count += inbag_samples_sorted[j, i]
                    CDF_by_tree[i, j:] += inbag_samples_sorted[j, i]

        CDF_by_tree[i, :] /= count

    return CDF_by_tree.sum(axis=0) / n_trees


dict_forest_key_type = nb.types.UniTuple(nb.types.int64, NUMBERS_OF_TREES)
dict_forest_value_type = nb.types.float64[:]
@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:])", nogil=True, cache=False, parallel=False)
def _compute_conditional_quantile_with_bootstrap_data_7(output_samples, samples_nodes, inbag_samples, X_nodes, alpha):
    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_quantiles = X_nodes.shape[0]
    n_alphas = alpha.shape[0]
    quantiles = np.zeros((n_quantiles, n_alphas), dtype=np.float64)
    
    order_statistic = np.argsort(output_samples)
    samples_nodes_sorted = np.asfortranarray(samples_nodes[order_statistic])
    inbag_samples_sorted = np.asfortranarray(inbag_samples[order_statistic])

    forest_cache = nb.typed.Dict.empty(key_type=dict_forest_key_type, value_type=dict_forest_value_type)
    CDF_by_tree = np.empty((n_trees, n_samples), dtype=np.float64)

    for i in range(n_quantiles):
        key_forest = to_fixed_tuple(X_nodes[i,:], NUMBERS_OF_TREES)
        if not bool(key_forest in forest_cache):
            CDF_forest = conditional_CDF_5(CDF_by_tree,
                                           samples_nodes_sorted,
                                           inbag_samples_sorted,
                                           X_nodes[i,:])
    
            quantiles[i, :] = np.array([
                                output_samples[
                                    order_statistic[
                                        np.argmax((CDF_forest >= alpha_var).astype(np.uint32))]
                                            ] 
                                for alpha_var in alpha])
            forest_cache[key_forest] = quantiles[i, :].copy()
        else:
            quantiles[i, :] = forest_cache[key_forest]
    return quantiles


# -----------------------------------------------------------------------------
# SEVENTH CODE
# -----------------------------------------------------------------------------

@nb.njit("float64[:](float64[:,:], DictType(UniTuple(int64, 2), float64[:]),int64[:,:], uint32[:,:], int64[:])", nogil=True, cache=False, parallel=True)
def conditional_CDF_6(CDF_by_tree, cache, samples_nodes_sorted, inbag_samples_sorted, X_nodes):
    n_samples = samples_nodes_sorted.shape[0]
    n_trees = samples_nodes_sorted.shape[1]
    
    for i in nb.prange(n_trees):
        key_tree = (X_nodes[i], nb.types.int64(i))
        if not bool(key_tree in cache):
            count = 0.
            CDF_by_tree[i, :] = 0.
            for j in range(n_samples):
                if inbag_samples_sorted[j, i] != 0:
                    if samples_nodes_sorted[j, i] == X_nodes[i]:
                        count += inbag_samples_sorted[j, i]
                        CDF_by_tree[i, j:] += inbag_samples_sorted[j, i]

            CDF_by_tree[i, :] /= count
        else:
            CDF_by_tree[i, :] = cache[key_tree]

    return CDF_by_tree.sum(axis=0) / n_trees

dict_key_type = nb.types.UniTuple(nb.types.int64, 2)
dict_value_type = nb.types.float64[:]
@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], int64[:,:], int64[:], float64[:], int64)", nogil=True, cache=False, parallel=False)
def _compute_conditional_quantile_with_bootstrap_data_8(output_samples, 
                                                        samples_nodes,
                                                        inbag_samples,
                                                        X_nodes,
                                                        idx_unique_X_nodes,
                                                        alpha,
                                                        max_cache_length):
    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    n_alphas = alpha.shape[0]
    quantiles = np.zeros((idx_unique_X_nodes.shape[0], n_alphas), dtype=np.float64)
    
    order_statistic = np.argsort(output_samples)
    samples_nodes_sorted = np.asfortranarray(samples_nodes[order_statistic])
    inbag_samples_sorted = np.asfortranarray(inbag_samples[order_statistic])

    tree_cache = nb.typed.Dict.empty(key_type=dict_key_type, value_type=dict_value_type)
    CDF_by_tree = np.empty((n_trees, n_samples), dtype=np.float64)

    for i in range(idx_unique_X_nodes.size):
        if len(tree_cache) > max_cache_length:
            for j in range(len(tree_cache) - max_cache_length):
                tree_cache.popitem()
                
        CDF_forest = conditional_CDF_6(CDF_by_tree,
                                       tree_cache,
                                       samples_nodes_sorted,
                                       inbag_samples_sorted,
                                       X_nodes[idx_unique_X_nodes[i],:])
        
        for k in range(n_trees):
            key_tree = (X_nodes[idx_unique_X_nodes[i], k], k)
            if not bool(key_tree in tree_cache):
                tree_cache[key_tree] = CDF_by_tree[k, :].copy()
            
        quantiles[i, :] = np.array([
                            output_samples[
                                order_statistic[
                                    np.argmax((CDF_forest >= alpha_var).astype(np.uint32))]
                                        ] 
                            for alpha_var in alpha])

    return quantiles


# -----------------------------------------------------------------------------
# EIGHTH CODE
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


@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:])", nogil=True, cache=False, parallel=False)
def _compute_conditional_quantile_with_bootstrap_data_9(output_samples, 
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

    CDF_by_tree = np.empty((n_samples), dtype=np.float64)
    CDF_forest = np.empty((n_quantiles, n_samples), dtype=np.float64)
    
    for j in range(n_trees):
        last_value = -1
        for i in np.argsort(X_nodes[:, j]):
            if last_value != X_nodes[i, j]:
                last_value = X_nodes[i, j]
                conditional_CDF_7(CDF_by_tree,
                                  j,
                                  samples_nodes_sorted,
                                  inbag_samples_sorted,
                                  last_value)
            
            CDF_forest[i, :] += CDF_by_tree
                
    for i in range(n_quantiles):
        CDF_forest[i,:] /= n_trees
        quantiles[i,:] = np.array([
                        output_samples[
                            order_statistic[
                                np.argmax((CDF_forest[i,:] >= alpha_var).astype(np.uint32))]
                                    ] 
                        for alpha_var in alpha])

    return quantiles


# -----------------------------------------------------------------------------
# NINTH CODE
# -----------------------------------------------------------------------------
  
@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:])", nogil=True, cache=False, parallel=True)
def _compute_conditional_quantile_with_bootstrap_data_10(output_samples,
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
        # Sorted X nodes
        idx_X_nodes_sorted = np.argsort(X_nodes[:, j])
        
        # Unique X_nodes with corresponding index in the previous result of argsort
        idx_X_nodes_unique = np.empty((n_quantiles+1), dtype=np.int64)
        idx_X_nodes_unique[0] = 0
        size_idx_X_nodes_unique = 1
        for k in range(1, n_quantiles):
            # Check if X_nodes change
            if X_nodes[idx_X_nodes_sorted[k], j] != X_nodes[idx_X_nodes_sorted[k-1], j]:
                idx_X_nodes_unique[size_idx_X_nodes_unique] = k
                size_idx_X_nodes_unique += 1
        idx_X_nodes_unique[size_idx_X_nodes_unique] = n_quantiles
        # No need to increment the size, the last value is for completeness
        
        # Distributed calculation of all unique values
        for i in nb.prange(size_idx_X_nodes_unique):

            # Range of unique copies of the current X_node
            first_pos = idx_X_nodes_unique[i]
            last_pos = idx_X_nodes_unique[i+1]            
            X_node = X_nodes[idx_X_nodes_sorted[first_pos], j]
            
            # Calculation for this X_node
            CDF_by_tree = np.empty((n_samples), dtype=np.float64)
            conditional_CDF_7(CDF_by_tree,
                              j,
                              samples_nodes_sorted,
                              inbag_samples_sorted,
                              X_node)
            
            for k in range(first_pos, last_pos):
                CDF_forest[idx_X_nodes_sorted[k], :] += CDF_by_tree
    
    for i in nb.prange(n_quantiles):
        CDF_forest[i,:] /= n_trees
        quantiles[i,:] = np.array([
                        output_samples[
                            order_statistic[
                                np.argmax((CDF_forest[i,:] >= alpha_var).astype(np.uint32))]
                                    ] 
                        for alpha_var in alpha])

    return quantiles


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
  
@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:])", nogil=True, cache=False, parallel=True)
def _compute_conditional_quantile_with_bootstrap_data_11(output_samples, 
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

    CDF_forest = np.zeros((n_quantiles, n_samples), dtype=np.float64)
    
    for j in nb.prange(n_trees):
        CDF_by_tree = np.empty((n_quantiles, n_samples), dtype=np.float64)
        last_value = -1
        argsort_X_nodes = np.argsort(X_nodes[:, j])
        for k in range(argsort_X_nodes.size):
            i = argsort_X_nodes[k]
            if last_value != X_nodes[i,j]:
                last_value = X_nodes[i, j]
                conditional_CDF_7(CDF_by_tree[i, :],
                                  j,
                                  samples_nodes_sorted,
                                  inbag_samples_sorted,
                                  last_value)
            else:
                CDF_by_tree[i, :] = CDF_by_tree[argsort_X_nodes[k-1], :]
        CDF_forest += CDF_by_tree
                
    for i in range(n_quantiles):
        CDF_forest[i,:] /= n_trees
        quantiles[i,:] = np.array([
                        output_samples[
                            order_statistic[
                                np.argmax((CDF_forest[i,:] >= alpha_var).astype(np.uint32))]
                                    ] 
                        for alpha_var in alpha])

    return quantiles

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

n_samples = 5*10**4
n_trees = NUMBERS_OF_TREES

max_cache_length = np.int64(0.8 * psutil.virtual_memory().available / (8 * n_samples))
print(f"Maximum number of cache entries: {max_cache_length}")

# Size of the leaf nodes
n_min_leaf = 20
n_min_split = n_min_leaf*2

# Use one sample to calibrate the forest, the quantiles and the indices
X = np.random.exponential(size = (n_samples, 2))
Y = X[:,0] - X[:,1]
X1 = X[:,0].reshape(-1,1)

quantForest = RandomForestRegressor(n_estimators=n_trees, min_samples_split=n_min_split, min_samples_leaf=n_min_leaf, n_jobs=-1)
quantForest.fit(X1, Y)

samples_nodes = quantForest.apply(X1)
output_samples = Y
X_nodes_val = np.random.exponential(size = (n_samples, 1))    
X_nodes = quantForest.apply(X_nodes_val)
alpha = np.array([0.3, 0.5, 0.9])

idx_bootstrap_samples, inbag_samples = compute_inbag_samples(n_samples, quantForest)
n_leaf = np.int64(node_count_forest(quantForest))
print('The number of count is ', n_leaf)

#start = time.time()
#for i in range(1):
#    a = _compute_conditional_quantile_with_bootstrap_data_4(n_leaf, output_samples, samples_nodes, inbag_samples, X_nodes, alpha)
#print('Time for the third : ', time.time() - start, a.mean())
#
#start = time.time()
#for i in range(1):
#    b = _compute_conditional_quantile_with_bootstrap_data_5(output_samples, samples_nodes, inbag_samples, X_nodes, alpha, max_cache_length)
#print('Time for the fourth : ', time.time() - start, b.mean())
#
#start = time.time()
#for i in range(5):
#    c = _compute_conditional_quantile_with_bootstrap_data_6(output_samples, samples_nodes, inbag_samples, X_nodes, alpha, max_cache_length)
#print('Time for the fifth : ', time.time() - start)
#
#start = time.time()
#for i in range(5):
#    d = _compute_conditional_quantile_with_bootstrap_data_7(output_samples, samples_nodes, inbag_samples, X_nodes, alpha)
#print('Time for the sixth : ', time.time() - start)
#
#start = time.time()
#_, ind, rev_ind = np.unique(X_nodes, axis=0, return_index=True, return_inverse=True)
#for i in range(5):
#    e = _compute_conditional_quantile_with_bootstrap_data_8(output_samples, samples_nodes, inbag_samples, X_nodes, ind, alpha, max_cache_length)
#    e = e[rev_ind]
#print('Time for the seventh : ', time.time() - start)
#
#start = time.time()
#for i in range(5):
#   f = _compute_conditional_quantile_with_bootstrap_data_9(output_samples, samples_nodes, inbag_samples, X_nodes, alpha)
#print('Time for the eighth : ', time.time() - start)

start = time.time()
_, ind, rev_ind = np.unique(X_nodes, axis=0, return_index=True, return_inverse=True)
for i in range(10):
   g = _compute_conditional_quantile_with_bootstrap_data_10(output_samples, samples_nodes, inbag_samples, X_nodes[ind], alpha)
   g = g[rev_ind]
print('Time for the ninth : ', time.time() - start)

start = time.time()
_, ind, rev_ind = np.unique(X_nodes, axis=0, return_index=True, return_inverse=True)
for i in range(10):
   h = _compute_conditional_quantile_with_bootstrap_data_10_bis(output_samples, samples_nodes, inbag_samples, X_nodes[ind], alpha)
   h = h[rev_ind]
print('Time for the ninth bis: ', time.time() - start)

#start = time.time()
#for i in range(5):
#   h = _compute_conditional_quantile_with_bootstrap_data_11(output_samples, samples_nodes, inbag_samples, X_nodes, alpha)
#print('Time for the tenth : ', time.time() - start)

#print((a != b).sum(axis=0))
#print((a != c).sum(axis=0))
#print((b != c).sum(axis=0))
#print((a != d).sum(axis=0))
#print((b != d).sum(axis=0))
#print((c != d).sum(axis=0))
#print((e != d).sum(axis=0))
#print((e != f).sum(axis=0))
#print((e != g).sum(axis=0))
print((g != h).sum(axis=0))