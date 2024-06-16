# -*- coding: utf-8 -*-

import cProfile
import numba as nb
import numpy as np
import pstats
import time
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor


# -------------------
# !!!!!!!!!!!!!!!!!!!
# -------------------
#
# Ancillary functions
#
# -------------------
# !!!!!!!!!!!!!!!!!!!
# -------------------

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


@nb.njit("float64[:](float64[:], int64[:,:], int64[:,:], int64[:], float64[:])", nogil=True, cache=False, parallel=False)
def _compute_conditional_minimum_expectation_with_weights_of_complete_forest_2(output_samples,
                                                                               samples_nodes,
                                                                               unique_X2_temp_nodes,
                                                                               counts_unique_X2_temp_nodes,
                                                                               alpha):
    n_alphas = alpha.shape[0]    
    n_samples = output_samples.shape[0]
    n_unique_X2_temp_nodes = unique_X2_temp_nodes.shape[0]
    argsort_output_samples = np.argsort(output_samples)
    mean_weights = np.zeros(n_samples, dtype=np.float64)
    minimum_expectation = np.empty(n_alphas, dtype=np.float64)
    
    # For each observation, complete the averaged weight thanks to the forest built with all variables
    for i in range(n_unique_X2_temp_nodes):
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

seed = 888
np.random.seed(seed)

# Parameters
n_samples = 10**4
n_trees = 10**2
min_samples_leaf = 10
alpha = np.array([0.1, 0.5, 0.7])
n_alphas = alpha.shape[0]
dim = 2

# Samples
X1 = np.random.exponential(size = (n_samples, 2))
Y1 = X1[:,0] - X1[:,1]
X2 = np.random.exponential(size = (n_samples, 2))

# Construction of the forest
forest = RandomForestRegressor(n_estimators=n_trees,
                               min_samples_leaf=min_samples_leaf,
                               min_samples_split=min_samples_leaf*2,
                               random_state=seed,
                               n_jobs=-1)
forest.fit(X1, Y1)

output_samples = Y1
samples_nodes = forest.apply(X1) # Get the leaf nodes of each observation in each tree
forest.n_jobs = 1

def _compute_numerator_qosa_index_by_variable(X2, feature, idx_min_obs, 
                                              forest,
                                              output_samples,
                                              samples_nodes,
                                              alpha):
    X2_temp = X2.copy()
    X2_temp[:, feature] = X2[idx_min_obs, feature]
    
    X2_temp_nodes = forest.apply(X2_temp)
    _, idx_unique_X2_temp_nodes, counts_unique_X2_temp_nodes = np.unique(
                                                                     X2_temp_nodes, 
                                                                     axis=0,
                                                                     return_index=True,
                                                                     return_inverse=False,
                                                                     return_counts=True)
        
    minimum_expectation = _compute_conditional_minimum_expectation_with_weights_of_complete_forest_2(
                                                    output_samples,
                                                    samples_nodes,
                                                    X2_temp_nodes[idx_unique_X2_temp_nodes],
                                                    counts_unique_X2_temp_nodes,
                                                    alpha)
    return minimum_expectation


# Compute the numerator of the indices for each variable
numerator = np.empty((dim, n_alphas), dtype=np.float64)
start = time.time()
parallel = Parallel(n_jobs=-1, batch_size=10, mmap_mode='r+')
# profiler = cProfile.Profile(builtins = False)
# profiler.enable()
for i in range(dim):
    print('variable : ', i)
    
    results = parallel(delayed(_compute_numerator_qosa_index_by_variable)(X2, i, j, forest,
                                                                output_samples,
                                                                samples_nodes,
                                                                alpha) for j in range(n_samples))
    minimum_expectation = np.asarray(results)    
    numerator[i, :] = minimum_expectation.mean(axis=0)

# profiler.disable()
# profiler.dump_stats('normal_profiler')
# pstats.Stats('normal_profiler').strip_dirs().sort_stats('time').print_stats(20)
print('Elapsed time with processes :', time.time()-start)       


# Compute the numerator of the indices for each variable
numerator = np.empty((dim, n_alphas), dtype=np.float64)
start = time.time()
parallel = Parallel(n_jobs=-1, backend="threading")
# profiler = cProfile.Profile(builtins = False)
# profiler.enable()
for i in range(dim):
    print('variable : ', i)
    
    results = parallel(delayed(_compute_numerator_qosa_index_by_variable)(X2, i, j, forest,
                                                                output_samples,
                                                                samples_nodes,
                                                                alpha) for j in range(n_samples))
    minimum_expectation = np.asarray(results)    
    numerator[i, :] = minimum_expectation.mean(axis=0)

# profiler.disable()
# profiler.dump_stats('normal_profiler')
# pstats.Stats('normal_profiler').strip_dirs().sort_stats('time').print_stats(20)
print('Elapsed time with threads:', time.time()-start)           


     