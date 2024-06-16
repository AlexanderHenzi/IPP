# -*- coding: utf-8 -*-

import time
import numba as nb
import numpy as np
from qosa import QuantileRegressionForest
from warnings import warn

# -----------------------------------------------------------------------------
#
# Implementation of the OOB error
#
# -----------------------------------------------------------------------------

# n_samples = 10**3
# n_trees = 10**1

# # Size of the leaf nodes
# n_min_leaf = 20
# n_min_split = n_min_leaf*2

# # Use one sample to calibrate the forest, the quantiles and the indices
# X = np.random.exponential(size = (n_samples, 2))
# Y = X[:,0] - X[:,1]
# X1 = X[:,0].reshape(-1,1)
# alpha = np.array([0.3, 0.5, 0.9])

# qrf= QuantileRegressionForest(n_estimators=n_trees, 
#                               min_samples_split=n_min_split, 
#                               min_samples_leaf=n_min_leaf, 
#                               n_jobs=-1)
# qrf.fit(X1, Y, oob_score_quantile=True, alpha=alpha, used_bootstrap_samples=False)

# samples_nodes = qrf.apply(X1)
# output_samples = Y

# idx_bootstrap_samples, inbag_samples = qrf._compute_inbag_samples()


# @nb.njit("float64[:](float64[:,:], float64[:])", nogil=True, cache=False, parallel=True)
# def _averaged_check_function_alpha_array(u, alpha):
#     """
#     Definition of the check function also called pinball loss function.
#     """
#     n_alphas = alpha.shape[0]
#     n_samples = u.shape[0]

#     check_function = np.empty((n_alphas, n_samples), dtype=np.float64)
#     for i in nb.prange(n_samples):
#         for j in range(n_alphas):
#             check_function[j,i] = u[i,j]*(alpha[j] - (u[i,j] < 0.))

#     averaged_check_function = np.empty((n_alphas), dtype=np.float64)
#     for i in nb.prange(n_alphas):
#         averaged_check_function[i] = check_function[i,:].mean()

#     return averaged_check_function

# from sklearn.ensemble.forest import _generate_unsampled_indices

# def compute_oob_samples(forest):
#     idx_oob_samples = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.int32[:])
#     n_trees = forest.n_estimators
#     n_samples = forest._n_samples  
    
#     # Here at each iteration we obtain out of bag samples for every tree.
#     for i in range(n_trees):
#         idx_oob_samples.update({i : _generate_unsampled_indices(forest.estimators_[i].random_state, 
#                                                                 n_samples,
#                                                                 n_samples)})
        
#     return idx_oob_samples

# @nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], DictType(int64, int32[:]), float64[:])", nogil=True, cache=False, parallel=True)
# def _compute_conditional_quantile_for_oob_samples_on_each_tree( 
#                                                        output_samples,
#                                                        samples_nodes,
#                                                        idx_bootstrap_samples,
#                                                        idx_oob_samples,
#                                                        alpha):
#     n_samples = samples_nodes.shape[0]
#     n_trees = samples_nodes.shape[1]
#     n_alphas = alpha.shape[0]

#     # Unique leaf nodes of each tree
#     samples_nodes = np.asfortranarray(samples_nodes)
#     idx_leaves_by_tree = [np.unique(samples_nodes[:, i]) for i in range(n_trees)]
    
#     quantiles = np.zeros((n_trees, n_samples, n_alphas), dtype=np.float64)
#     n_quantiles = np.zeros((n_trees, n_samples), dtype=np.uint32)
    
#     for i in nb.prange(n_trees):
#         idx_leaves_for_tree_i = idx_leaves_by_tree[i]
#         X_nodes_oob_for_tree_i = samples_nodes[:, i][idx_oob_samples[np.int64(i)]]
#         n_X_nodes_oob_for_tree_i = X_nodes_oob_for_tree_i.shape[0]
#         Y_leaves = np.empty((n_samples), dtype=np.float64)
#         idx_X_nodes_oob_in_current_leaf = np.empty((n_X_nodes_oob_for_tree_i), dtype=np.uint32)
        
#         for idx_current_leaf in idx_leaves_for_tree_i:            
#             n_idx_X_nodes_oob_in_current_leaf = 0
#             for j in range(n_X_nodes_oob_for_tree_i):
#                 if X_nodes_oob_for_tree_i[j] == idx_current_leaf:
#                     idx_X_nodes_oob_in_current_leaf[n_idx_X_nodes_oob_in_current_leaf] = j
#                     n_idx_X_nodes_oob_in_current_leaf += 1
            
#             n_Y_leaves = 0
#             for k in range(n_samples):
#                 if samples_nodes[idx_bootstrap_samples[k, i], i] == idx_current_leaf:
#                     Y_leaves[n_Y_leaves] = output_samples[idx_bootstrap_samples[k, i]]
#                     n_Y_leaves += 1
                        
#             quantiles_in_leaf = np.percentile(Y_leaves[:np.int64(n_Y_leaves)], alpha*100)            
#             for l in range(n_idx_X_nodes_oob_in_current_leaf):
#                 quantiles[i, idx_oob_samples[np.int64(i)][idx_X_nodes_oob_in_current_leaf[l]], :quantiles.shape[2]] = quantiles_in_leaf
#                 n_quantiles[i, idx_oob_samples[np.int64(i)][idx_X_nodes_oob_in_current_leaf[l]]] = 1
    
#     n_quantiles = n_quantiles.sum(axis=0)
#     n_quantiles[n_quantiles == 0] = 1
    
#     return quantiles.sum(axis=0)/n_quantiles.reshape(-1,1)

# idx_oob_samples = compute_oob_samples(qrf)

# def _set_oob_score(self, X, y):
#    """
#    Compute out-of-bag scores.
#    """

#    X = self._input_samples
#    y = self._output_samples

#    n_samples = self._n_samples
   
#    idx_oob_samples = compute_oob_samples(self)
   
#    conditional_quantiles =_compute_conditional_quantile_for_oob_samples_on_each_tree( 
#                                                            output_samples,
#                                                            samples_nodes,
#                                                            idx_bootstrap_samples,
#                                                            idx_oob_samples,
#                                                            alpha)

#    if (conditional_quantiles.sum(axis=1) == 0).any():
#        warn("Some inputs do not have OOB scores. "
#             "This probably means too few trees were used "
#             "to compute any reliable oob estimates.")

#    self.oob_prediction_ = conditional_quantiles
#    self.oob_score_ = _averaged_check_function_alpha_array(
#                                         y.reshape(-1,1)-conditional_quantiles,
#                                         alpha)
   
   
   
# @nb.njit("float64[:,:](float64[:], int64[:,:], DictType(int64, int32[:]), float64[:])", nogil=True, cache=False, parallel=True)
# def _compute_conditional_quantile_for_oob_samples_on_each_tree_with_original_data( 
#                                                        output_samples,
#                                                        samples_nodes,
#                                                        idx_oob_samples,
#                                                        alpha):
#     n_samples = samples_nodes.shape[0]
#     n_trees = samples_nodes.shape[1]
#     n_alphas = alpha.shape[0]

#     # Unique leaf nodes of each tree
#     samples_nodes = np.asfortranarray(samples_nodes)
#     idx_leaves_by_tree = [np.unique(samples_nodes[:, i]) for i in range(n_trees)]
    
#     quantiles = np.zeros((n_trees, n_samples, n_alphas), dtype=np.float64)
#     n_quantiles = np.zeros((n_trees, n_samples), dtype=np.uint32)
    
#     for i in nb.prange(n_trees):
#         idx_leaves_for_tree_i = idx_leaves_by_tree[i]
#         X_nodes_oob_for_tree_i = samples_nodes[:, i][idx_oob_samples[np.int64(i)]]
#         n_X_nodes_oob_for_tree_i = X_nodes_oob_for_tree_i.shape[0]
#         idx_Y_in_current_leaf = np.empty((n_samples), dtype=np.uint32)
#         idx_X_nodes_oob_in_current_leaf = np.empty((n_X_nodes_oob_for_tree_i), dtype=np.uint32)
        
#         for idx_current_leaf in idx_leaves_for_tree_i:            
#             n_idx_X_nodes_oob_in_current_leaf = 0
#             for j in range(n_X_nodes_oob_for_tree_i):
#                 if X_nodes_oob_for_tree_i[j] == idx_current_leaf:
#                     idx_X_nodes_oob_in_current_leaf[n_idx_X_nodes_oob_in_current_leaf] = j
#                     n_idx_X_nodes_oob_in_current_leaf += 1
            
#             n_idx_Y_in_current_leaf = 0
#             for k in range(n_samples):
#                 if samples_nodes[k, i] == idx_current_leaf:
#                     idx_Y_in_current_leaf[n_idx_Y_in_current_leaf] = k
#                     n_idx_Y_in_current_leaf += 1
                        
#             for l in range(n_idx_X_nodes_oob_in_current_leaf):
#                 quantiles_in_leaf = np.percentile(
#                     output_samples[
#                         idx_Y_in_current_leaf[:np.int64(n_idx_Y_in_current_leaf)][
#                             idx_Y_in_current_leaf[:np.int64(n_idx_Y_in_current_leaf)] != 
#                             idx_oob_samples[np.int64(i)][idx_X_nodes_oob_in_current_leaf[l]]]
#                         ],
#                     alpha*100)
#                 quantiles[i, 
#                           idx_oob_samples[np.int64(i)][idx_X_nodes_oob_in_current_leaf[l]],
#                           :quantiles.shape[2]] = quantiles_in_leaf
#                 n_quantiles[i, idx_oob_samples[np.int64(i)][idx_X_nodes_oob_in_current_leaf[l]]] = 1
    
#     n_quantiles = n_quantiles.sum(axis=0)
#     n_quantiles[n_quantiles == 0] = 1
    
#     return quantiles.sum(axis=0)/n_quantiles.reshape(-1,1)

# conditional_quantiles = _compute_conditional_quantile_for_oob_samples_on_each_tree_with_original_data( 
#                                                        output_samples,
#                                                        samples_nodes,
#                                                        idx_oob_samples,
#                                                        alpha)


# -----------------------------------------------------------------------------
#
# Implementation of the OOB error for local average estimator of the CCDF
#
# -----------------------------------------------------------------------------

n_samples = 10**4
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
qrf.fit(X1, Y, oob_score_quantile=True, alpha=alpha, method="Averaged_Quantile", used_bootstrap_samples=False)
qrf.fit(X1, Y, oob_score_quantile=True, alpha=alpha, method="Weighted_CDF", used_bootstrap_samples=False)

qrf.fit(X1, Y)
samples_nodes = qrf.apply(X1)
output_samples = Y
_, inbag_samples = qrf._compute_inbag_samples()




@nb.njit("float64[:](int64[:,:], uint32[:,:], int64[:])", nogil=True, cache=True, parallel=False)
def _compute_weight_with_bootstrap_data(samples_nodes, inbag_samples, X_nodes_k):
    """
    Function to compute the bootstrapped averaged weight of each individual of
    the original training sample in the forest.
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




@nb.njit("float64[:](int64[:,:], int64[:])", nogil=True, cache=True, parallel=False)
def _compute_weight_with_original_data(samples_nodes, X_nodes_k):
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




@nb.njit("void(float64[:,:], float64[:], int64[:,:], uint32[:,:], float64[:])", nogil=True, cache=True, parallel=True)
def _compute_conditional_quantile_for_oob_samples_with_Weighted_CDF_using_bootstrap_data(quantiles,
                                                                                         output_samples,
                                                                                         samples_nodes,
                                                                                         inbag_samples,
                                                                                         alpha):
    """
    Predictions and scores OOB by using the weights based on the bootstrap samples 
    for calculating the conditional quantiles.
    """
    
    n_samples = samples_nodes.shape[0]
    order_statistic = np.argsort(output_samples)
    
    # Calculate the conditional quantile of each observation of the original sample
    for i in nb.prange(n_samples):
        trees_built_without_obs_i_in_bootstrap_samples = np.where(inbag_samples[i] == 0)[0]
        
        if trees_built_without_obs_i_in_bootstrap_samples.size != 0:
            
            # Compute the Boostrap Data based conditional weights associated to each individual
            weight = _compute_weight_with_bootstrap_data(samples_nodes[:, trees_built_without_obs_i_in_bootstrap_samples],
                                                         inbag_samples[:, trees_built_without_obs_i_in_bootstrap_samples],
                                                         samples_nodes[i][trees_built_without_obs_i_in_bootstrap_samples])
            
            # Compute the quantiles thanks to the Cumulative Distribution Function 
            # for each value of alpha
            CDF = np.cumsum(weight[order_statistic])
            
            # ":quantiles.shape[1]" : quantiles.shape[1] necessary because of a bug of numba combined with parallel=True and the condition if
            quantiles[i, :quantiles.shape[1]] = np.array([
                output_samples[
                    order_statistic[
                        np.argmax((CDF >= alpha_var).astype(np.uint32))]
                    ]
                for alpha_var in alpha])

        


@nb.njit("void(float64[:,:], float64[:], int64[:,:], uint32[:,:], float64[:])", nogil=True, cache=True, parallel=True)
def _compute_conditional_quantile_for_oob_samples_with_Weighted_CDF_using_original_data(quantiles,
                                                                                        output_samples,
                                                                                        samples_nodes,
                                                                                        inbag_samples,
                                                                                        alpha):
    """
    Predictions and scores OOB by using the weights based on the original sample
    for calculating the conditional quantiles.
    """
    
    n_samples = samples_nodes.shape[0]
    idx_n_samples = np.arange(0, n_samples)
    
    # Calculate the conditional quantile of each observation of the original sample
    for i in nb.prange(n_samples):
        trees_built_without_obs_i_in_bootstrap_samples = np.where(inbag_samples[i] == 0)[0]
        
        if trees_built_without_obs_i_in_bootstrap_samples.size != 0:
            
            idx_n_samples_without_original_obs_i = np.delete(idx_n_samples, i)
            order_statistic = np.argsort(output_samples[idx_n_samples_without_original_obs_i])
            
            # Compute the Original Data based conditional weights associated to each individual
            weight = _compute_weight_with_original_data(
                samples_nodes[:, trees_built_without_obs_i_in_bootstrap_samples][idx_n_samples_without_original_obs_i],
                samples_nodes[i][trees_built_without_obs_i_in_bootstrap_samples])
            
            # Compute the quantiles thanks to the Cumulative Distribution Function 
            # for each value of alpha
            CDF = np.cumsum(weight[order_statistic])
            
            # ":quantiles.shape[1]" : quantiles.shape[1] necessary because of a bug of numba combined with parallel=True and the condition if
            quantiles[i, :quantiles.shape[1]] = np.array([
                output_samples[idx_n_samples_without_original_obs_i][
                    order_statistic[
                        np.argmax((CDF >= alpha_var).astype(np.uint32))]
                    ]
                for alpha_var in alpha])
            
            
            
@nb.njit("void(float64[:,:], float64[:], int64[:,:], uint32[:,:], float64[:])", nogil=True, cache=True, parallel=True)
def _compute_conditional_quantile_for_oob_samples_with_Weighted_CDF_using_original_data_2(quantiles,
                                                                                          output_samples,
                                                                                          samples_nodes,
                                                                                          inbag_samples,
                                                                                          alpha):
    """
    Predictions and scores OOB by using the weights based on the original sample
    for calculating the conditional quantiles.
    """
    
    n_samples = samples_nodes.shape[0]
    idx_n_samples = np.arange(0, n_samples)
    order_statistic = np.argsort(output_samples)
    
    # Calculate the conditional quantile of each observation of the original sample with the OOB samples
    for i in nb.prange(n_samples):
        trees_built_without_obs_i_in_bootstrap_samples = np.where(inbag_samples[i] == 0)[0]
        
        if trees_built_without_obs_i_in_bootstrap_samples.size != 0:
            
            idx_n_samples_without_original_obs_i = np.delete(idx_n_samples, i)
            
            # Compute the Original Data based conditional weights associated to each individual
            weight = _compute_weight_with_original_data(
                samples_nodes[:, trees_built_without_obs_i_in_bootstrap_samples][idx_n_samples_without_original_obs_i],
                samples_nodes[i][trees_built_without_obs_i_in_bootstrap_samples])
            
            # Compute the quantiles thanks to the Cumulative Distribution Function
            # for each value of alpha
            order_statistic_without_obs_i = order_statistic[order_statistic != i]
            order_statistic_without_obs_i[order_statistic_without_obs_i > i] = order_statistic_without_obs_i[
                order_statistic_without_obs_i > i] - 1
            CDF = np.cumsum(weight[order_statistic_without_obs_i])
            
            # ":quantiles.shape[1]" : quantiles.shape[1] necessary because of a bug of numba combined with parallel=True and the condition if
            quantiles[i, :quantiles.shape[1]] = np.array([
                output_samples[idx_n_samples_without_original_obs_i][
                    order_statistic_without_obs_i[
                        np.argmax((CDF >= alpha_var).astype(np.uint32))]
                    ]
                for alpha_var in alpha])
            
# start = time.perf_counter()
# quantiles_boot = np.zeros((n_samples, alpha.size),dtype=np.float64,order='C')
# _compute_conditional_quantile_for_oob_samples_with_Weighted_CDF_using_bootstrap_data(
#     quantiles_boot,
#     output_samples,
#     samples_nodes,
#     inbag_samples,
#     alpha)
# print("Total time = ", time.perf_counter() - start)

start = time.perf_counter()
quantiles_orig = np.zeros((n_samples, alpha.size),dtype=np.float64,order='C')
_compute_conditional_quantile_for_oob_samples_with_Weighted_CDF_using_original_data(
    quantiles_orig,
    output_samples,
    samples_nodes,
    inbag_samples,
    alpha)
print("Total time = ",time.perf_counter() - start)

start = time.perf_counter()
quantiles_orig_2 = np.zeros((n_samples, alpha.size),dtype=np.float64,order='C')
_compute_conditional_quantile_for_oob_samples_with_Weighted_CDF_using_original_data_2(
    quantiles_orig_2,
    output_samples,
    samples_nodes,
    inbag_samples,
    alpha)
print("Total time = ",time.perf_counter() - start)
