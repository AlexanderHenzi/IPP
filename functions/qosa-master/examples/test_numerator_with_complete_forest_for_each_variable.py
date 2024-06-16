# -*- coding: utf-8 -*-

import numba as nb
import numpy as np
import time
from joblib import Parallel, delayed

from qosa import MinimumConditionalExpectedCheckFunctionWithWeights
from qosa import qosa_Min__Weighted_Min_with_complete_forest, MinimumBasedQosaIndices


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


@nb.njit("float64[:](int64[:,:], uint32[:,:], int64[:])", nogil=True, cache=False, parallel=False)
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


@nb.njit("float64[:](int64[:,:], int64[:])", nogil=True, cache=False, parallel=False)
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


@nb.njit("float64(float64[:], float64[:], float64, float64)", nogil=True, cache=False, parallel=False)
def _averaged_check_function_alpha_float_unparallel_product_weight(output_samples, weight, theta, alpha):

    n_samples = output_samples.shape[0]
    res = 0.    
    for i in range(n_samples):
        res += (output_samples[i] - theta)*(alpha - ((output_samples[i] - theta) < 0.))*weight[i]
    
    return res


@nb.njit("void(float64[:], float64[:], int64[:,:], uint32[:,:], int64[:,:], int64[:], float64[:], boolean)", nogil=True, cache=False, parallel=False)
def _compute_conditional_minimum_expectation_with_weights_of_complete_forest(minimum_expectation,
                                                                             output_samples,
                                                                             samples_nodes,
                                                                             inbag_samples,
                                                                             unique_X2_temp_nodes,
                                                                             counts_unique_X2_temp_nodes,
                                                                             alpha,
                                                                             used_bootstrap_samples):
    
    n_samples = output_samples.shape[0]
    n_unique_X2_temp_nodes = unique_X2_temp_nodes.shape[0]
    argsort_output_samples = np.argsort(output_samples)
    mean_weights = np.zeros(n_samples, dtype=np.float64)

    # For each observation, complete the averaged weight thanks to the forest built with all variables
    for i in range(n_unique_X2_temp_nodes):
        if used_bootstrap_samples:
            # Compute the Boostrap Data based conditional weights associated to each individual
            weight = _compute_weight_with_bootstrap_data(samples_nodes,
                                                         inbag_samples,
                                                         unique_X2_temp_nodes[i, :])
        else:
            # Compute the Original Data based conditional weights associated to each individual
            weight = _compute_weight_with_original_data(samples_nodes, unique_X2_temp_nodes[i, :])
        
        mean_weights += weight*counts_unique_X2_temp_nodes[i]
    mean_weights /= n_samples
        
    k=0
    for j, alpha_temp in enumerate(alpha):
        if k < n_samples:
            ans = _averaged_check_function_alpha_float_unparallel_product_weight(
                                                 output_samples,
                                                 mean_weights,
                                                 output_samples[argsort_output_samples[k]],
                                                 alpha_temp)
        else:
            for l in range(j, alpha.shape[0]):
                minimum_expectation[l] = _averaged_check_function_alpha_float_unparallel_product_weight(
                                                     output_samples,
                                                     mean_weights,
                                                     output_samples[argsort_output_samples[-1]],
                                                     alpha[l])
            break
                
        if j == 0: 
            k=1
        else:
            k+=1
        while(k < n_samples):
            temp = _averaged_check_function_alpha_float_unparallel_product_weight(
                                             output_samples,
                                             mean_weights,
                                             output_samples[argsort_output_samples[k]],
                                             alpha_temp)
            if(temp <= ans):
                ans = temp
                k+=1
            else:
                k-=1
                break
        minimum_expectation[j] = ans


# @nb.njit(nb.float64[:](nb.float64[:], nb.types.Array(nb.types.int64, 2, 'C', readonly=True), nb.uint32[:,:], nb.int64[:,:], nb.int64[:], nb.float64[:], nb.boolean), nogil=True, cache=False, parallel=False)
@nb.njit("float64[:](float64[:], int64[:,:], uint32[:,:], int64[:,:], int64[:], float64[:], boolean)", nogil=True, cache=False, parallel=False)
def _compute_conditional_minimum_expectation_with_weights_of_complete_forest_2(
                                                                             output_samples,
                                                                             samples_nodes,
                                                                             inbag_samples,
                                                                             unique_X2_temp_nodes,
                                                                             counts_unique_X2_temp_nodes,
                                                                             alpha,
                                                                             used_bootstrap_samples):
    n_alphas = alpha.shape[0]    
    n_samples = output_samples.shape[0]
    n_unique_X2_temp_nodes = unique_X2_temp_nodes.shape[0]
    argsort_output_samples = np.argsort(output_samples)
    mean_weights = np.zeros(n_samples, dtype=np.float64)
    minimum_expectation = np.empty(n_alphas, dtype=np.float64)
    
    # For each observation, complete the averaged weight thanks to the forest built with all variables
    for i in range(n_unique_X2_temp_nodes):
        if used_bootstrap_samples:
            # Compute the Boostrap Data based conditional weights associated to each individual
            weight = _compute_weight_with_bootstrap_data(samples_nodes,
                                                         inbag_samples,
                                                         unique_X2_temp_nodes[i, :])
        else:
            # Compute the Original Data based conditional weights associated to each individual
            weight = _compute_weight_with_original_data(samples_nodes, unique_X2_temp_nodes[i, :])
        
        mean_weights += weight*counts_unique_X2_temp_nodes[i]
    mean_weights /= n_samples
    
    k=0
    for j, alpha_temp in enumerate(alpha):
        if k < n_samples:
            ans = _averaged_check_function_alpha_float_unparallel_product_weight(
                                                 output_samples,
                                                 mean_weights,
                                                 output_samples[argsort_output_samples[k]],
                                                 alpha_temp)
        else:
            for l in range(j, n_alphas):
                minimum_expectation[l] = _averaged_check_function_alpha_float_unparallel_product_weight(
                                                     output_samples,
                                                     mean_weights,
                                                     output_samples[argsort_output_samples[-1]],
                                                     alpha[l])
            break
                
        if j == 0: 
            k=1
        else:
            k+=1
        while(k < n_samples):
            temp = _averaged_check_function_alpha_float_unparallel_product_weight(
                                             output_samples,
                                             mean_weights,
                                             output_samples[argsort_output_samples[k]],
                                             alpha_temp)
            if(temp <= ans):
                ans = temp
                k+=1
            else:
                k-=1
                break
        minimum_expectation[j] = ans
    
    return minimum_expectation




# ------------------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ------------------------------------------------------------------------------------
#
# Compute the QOSA indices with the averaged weights computed with the complete forest
#
# ------------------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ------------------------------------------------------------------------------------

###########################
# Without parallelization #
###########################

# seed = 888
# np.random.seed(seed)

# n_samples = 10**3
# X1 = np.random.exponential(size = (n_samples, 2))
# Y1 = X1[:,0] - X1[:,1]

# # Second sample to estimate the index
# X2 = np.random.exponential(size = (n_samples, 2))

# dim = 2
# alpha = np.array([0.1, 0.5, 0.7])
# n_alphas = alpha.shape[0]
# n_estimators = 10**1
# min_samples_leaf = 10

# forest = MinimumConditionalExpectedCheckFunctionForest(n_estimators=n_estimators,
#                                                        min_samples_leaf=min_samples_leaf,
#                                                        min_samples_split=min_samples_leaf*2,
#                                                        random_state=seed,
#                                                        n_jobs=-1)
# forest.fit(X1,Y1)

# output_samples = Y1
# samples_nodes = forest._samples_nodes
# inbag_samples = np.empty((1, 1), dtype=np.uint32)
# used_bootstrap_samples = False

# # Compute the numerator of the indices for each variable
# numerator = np.empty((dim, n_alphas), dtype=np.float64)
# start = time.time()
# for i in range(dim):
#     print('variable : ', i)
#     minimum_expectation = np.empty((n_samples, n_alphas), 
#                                    dtype=np.float64,
#                                    order='C')
    
#     for j in range(n_samples):
#         X2_temp = X2.copy()
#         X2_temp[:, i] = X2[j, i]
        
#         X2_temp_nodes = forest.get_nodes(X2_temp)
#         _, idx_unique_X2_temp_nodes, counts_unique_X2_temp_nodes = np.unique(
#                                                                          X2_temp_nodes, 
#                                                                          axis=0,
#                                                                          return_index=True,
#                                                                          return_inverse=False,
#                                                                          return_counts=True)
        
#         _compute_conditional_minimum_expectation_with_weights_of_complete_forest(
#                                                         minimum_expectation[j, :],
#                                                         output_samples,
#                                                         samples_nodes,
#                                                         inbag_samples,
#                                                         X2_temp_nodes[idx_unique_X2_temp_nodes],
#                                                         counts_unique_X2_temp_nodes,
#                                                         alpha,
#                                                         used_bootstrap_samples)
    
#     numerator[i, :] = minimum_expectation.mean(axis=0)
# print('Elapsed time :', time.time()-start)       

# # Compute the denominator of the indices
# alpha_quantile = np.percentile(Y1, q=alpha*100)
# denominator = _averaged_check_function_alpha_array(Y1.reshape(-1,1) - alpha_quantile,
#                                                    alpha)

# qosa_indices = 1 - numerator/denominator
# qosa_indices.T


########################
# With parallelization #
########################

if __name__ == "__main__":
    seed = 888
    np.random.seed(seed)
    
    n_samples = 10**3
    X1 = np.random.exponential(size = (n_samples, 2))
    Y1 = X1[:,0] - X1[:,1]
    
    # Second sample to estimate the index
    X2 = np.random.exponential(size = (n_samples, 2))
    
    dim = 2
    alpha = np.array([0.1, 0.5, 0.7])
    n_alphas = alpha.shape[0]
    n_estimators = 10**1
    min_samples_leaf = 10
    
    forest = MinimumConditionalExpectedCheckFunctionWithWeights(n_estimators=n_estimators,
                                                                min_samples_leaf=min_samples_leaf,
                                                                min_samples_split=min_samples_leaf*2,
                                                                random_state=seed,
                                                                n_jobs=-1)
    forest.fit(X1,Y1)
    
    output_samples = Y1
    samples_nodes = forest._samples_nodes
    inbag_samples = np.empty((1, 1), dtype=np.uint32)
    used_bootstrap_samples = False
    forest.n_jobs = 1 # Back to serial execution of forest methods.
    
    def _compute_numerator_qosa_index_by_variable(X2, feature, idx_min_obs, 
                                                  forest,
                                                  output_samples,
                                                  samples_nodes,
                                                  inbag_samples,
                                                  alpha,
                                                  used_bootstrap_samples):
        X2_temp = X2.copy()
        X2_temp[:, feature] = X2[idx_min_obs, feature]
        
        X2_temp_nodes = forest.get_nodes(X2_temp)
        _, idx_unique_X2_temp_nodes, counts_unique_X2_temp_nodes = np.unique(
                                                                         X2_temp_nodes, 
                                                                         axis=0,
                                                                         return_index=True,
                                                                         return_inverse=False,
                                                                         return_counts=True)
            
        minimum_expectation = _compute_conditional_minimum_expectation_with_weights_of_complete_forest_2(
                                                        output_samples,
                                                        samples_nodes,
                                                        inbag_samples,
                                                        X2_temp_nodes[idx_unique_X2_temp_nodes],
                                                        counts_unique_X2_temp_nodes,
                                                        alpha,
                                                        used_bootstrap_samples)
        return minimum_expectation
        
    # Compute the numerator of the indices for each variable
    numerator = np.empty((dim, n_alphas), dtype=np.float64)
    start = time.time()
    parallel = Parallel(n_jobs=-1, mmap_mode='r+')
    # parallel = Parallel(n_jobs=-1, max_nbytes=None)
    for i in range(dim):
        print('variable : ', i)
        
        results = parallel(delayed(_compute_numerator_qosa_index_by_variable)(X2, i, j, forest,
                                                                    output_samples,
                                                                    samples_nodes,
                                                                    inbag_samples,
                                                                    alpha,
                                                                    used_bootstrap_samples) for j in range(n_samples))
        minimum_expectation = np.asarray(results)
        
        # for j in range(n_samples):
        #       _compute_numerator_qosa_index_by_variable(X2, i, j, forest,
        #                                           minimum_expectation,
        #                                           output_samples,
        #                                           samples_nodes,
        #                                           inbag_samples,
        #                                           alpha,
        #                                           used_bootstrap_samples)
        
        numerator[i, :] = minimum_expectation.mean(axis=0)
    print('Elapsed time :', time.time()-start)       
    
    # Compute the denominator of the indices
    alpha_quantile = np.percentile(Y1, q=alpha*100)
    denominator = _averaged_check_function_alpha_array(Y1.reshape(-1,1) - alpha_quantile,
                                                       alpha)
    
    qosa_indices_bis = 1 - numerator/denominator
    qosa_indices_bis.T
    
    print(qosa_indices_bis.T)
    # print(qosa_indices == qosa_indices_bis)
    
    
    print('je passe au pool')
    
    qosa = MinimumBasedQosaIndices()
    qosa.feed_sample(X1, Y1, "Weighted_Min_with_complete_forest", X2=X2)
    
    method = qosa_Min__Weighted_Min_with_complete_forest(alpha=alpha,
                                                         n_estimators=n_estimators,
                                                         min_samples_leaf=min_samples_leaf,
                                                         used_bootstrap_samples=False,
                                                         optim_by_CV=False,
                                                         n_fold=3,
                                                         random_state_Forest=seed)
    
    start = time.perf_counter()
    qosa_results = qosa.compute_indices(method)
    print("Total time = ",time.perf_counter() - start)
    
    print(qosa_results.qosa_indices_estimates)
    
    print(qosa_results.qosa_indices_estimates == qosa_indices_bis.T)