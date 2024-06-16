# -*- coding: utf-8 -*-

import numba as nb
import numpy as np
import time
from qosa import MinimumConditionalExpectedCheckFunctionForest


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

@nb.njit("void(float64[:,:], float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:], boolean)", nogil=True, cache=False, parallel=True)
def _compute_conditional_minimum_expectation(minimum_expectation,
                                             output_samples,
                                             samples_nodes,
                                             inbag_samples,
                                             X_nodes,
                                             alpha,
                                             used_bootstrap_samples):
    
    n_minimum_expectation = minimum_expectation.shape[0]
    argsort_output_samples = np.argsort(output_samples)

    # For each observation to compute
    for i in nb.prange(n_minimum_expectation):
        if used_bootstrap_samples:
            # Compute the Boostrap Data based conditional weights associated to each individual
            weight = _compute_weight_with_bootstrap_data(samples_nodes,
                                                         inbag_samples,
                                                         X_nodes[i, :])
        else:
            # Compute the Original Data based conditional weights associated to each individual
            weight = _compute_weight_with_original_data(samples_nodes, X_nodes[i, :])
        
        k=0
        for j, alpha_temp in enumerate(alpha):
            if k < output_samples.shape[0]:
                ans = _averaged_check_function_alpha_float_unparallel_product_weight(
                                                     output_samples,
                                                     weight,
                                                     output_samples[argsort_output_samples[k]],
                                                     alpha_temp)
            else:
                for l in range(j, alpha.shape[0]):
                    minimum_expectation[i, l] = _averaged_check_function_alpha_float_unparallel_product_weight(
                                                         output_samples,
                                                         weight,
                                                         output_samples[argsort_output_samples[-1]],
                                                         alpha[l])
                break
                    
            if j == 0: 
                k=1
            else:
                k+=1
            while(k < output_samples.shape[0]):
                temp = _averaged_check_function_alpha_float_unparallel_product_weight(
                                                 output_samples,
                                                 weight,
                                                 output_samples[argsort_output_samples[k]],
                                                 alpha_temp)
                if(temp <= ans):
                    ans = temp
                    k+=1
                else:
                    k-=1
                    break
            minimum_expectation[i, j] = ans


n_sample = 10**3
X = np.random.exponential(size = (n_sample, 2))
Y1 = X[:,0] - X[:,1]
X1 = X[:,0]

# Second sample to estimate the index
X2 = np.random.exponential(size = n_sample)

alpha = np.array([0.7, 0.99, 0.995])
n_alphas = alpha.shape[0]
n_estimators = 10**2
min_samples_leaf = 10

min_expectation = MinimumConditionalExpectedCheckFunctionForest(n_estimators=n_estimators,
                                                                min_samples_leaf=min_samples_leaf,
                                                                min_samples_split=min_samples_leaf*2)
min_expectation.fit(X1, Y1)

# ----------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------
#
# With the Original data
#
# ----------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------

used_bootstrap_samples = False

#################
# Normal method #
#################

start = time.time()
for i in range(10):
    minimum_expectation = min_expectation.predict(X=X2,
                                                  alpha=alpha,
                                                  used_bootstrap_samples=used_bootstrap_samples)
print('Elpased time: ', time.time()-start)

####################
# Optimized method #
####################

start = time.time()
for i in range(10):
    X_nodes = min_expectation.get_nodes(X2)
    _, idx_unique_X_nodes, idx_inverse_unique_X_nodes = np.unique(X_nodes, 
                                                                  axis=0,
                                                                  return_index=True,
                                                                  return_inverse=True)
    minimum_expectation_bis = np.empty((idx_unique_X_nodes.shape[0], n_alphas), dtype=np.float64)
    
    if used_bootstrap_samples:
        _, inbag_samples = min_expectation._compute_inbag_samples()
    else:
        inbag_samples = np.empty((1, 1), dtype=np.uint32)
    
    _compute_conditional_minimum_expectation(minimum_expectation_bis,
                                             Y1,
                                             min_expectation._samples_nodes,
                                             inbag_samples,
                                             X_nodes[idx_unique_X_nodes],
                                             alpha,
                                             used_bootstrap_samples)
    
    minimum_expectation_bis = minimum_expectation_bis[idx_inverse_unique_X_nodes]
print('Elpased time: ', time.time()-start)

print((minimum_expectation != minimum_expectation_bis).sum(axis=0))



# ----------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------
#
# With the bootstrap samples
#
# ----------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------

used_bootstrap_samples = True

#################
# Normal method #
#################

start = time.time()
for i in range(10):
    minimum_expectation = min_expectation.predict(X=X2,
                                                  alpha=alpha,
                                                  used_bootstrap_samples=used_bootstrap_samples)
print('Elpased time: ', time.time()-start)

####################
# Optimized method #
####################

start = time.time()
for i in range(10):
    X_nodes = min_expectation.get_nodes(X2)
    _, idx_unique_X_nodes, idx_inverse_unique_X_nodes = np.unique(X_nodes, 
                                                                  axis=0,
                                                                  return_index=True,
                                                                  return_inverse=True)
    minimum_expectation_bis = np.empty((idx_unique_X_nodes.shape[0], n_alphas), dtype=np.float64)
    
    if used_bootstrap_samples:
        _, inbag_samples = min_expectation._compute_inbag_samples()
    else:
        inbag_samples = np.empty((1, 1), dtype=np.uint32)
    
    _compute_conditional_minimum_expectation(minimum_expectation_bis,
                                             Y1,
                                             min_expectation._samples_nodes,
                                             inbag_samples,
                                             X_nodes[idx_unique_X_nodes],
                                             alpha,
                                             used_bootstrap_samples)
    
    minimum_expectation_bis = minimum_expectation_bis[idx_inverse_unique_X_nodes]
print('Elpased time: ', time.time()-start)

print((minimum_expectation != minimum_expectation_bis).sum(axis=0))