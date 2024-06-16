# -*- coding: utf-8 -*-


import time
import numba as nb
import numpy as np
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
                                                    n_samples)
            inbag_samples[:,idx_tree] = np.bincount(
                                                    idx_bootstrap_samples[:,idx_tree],
                                                    minlength=n_samples)
    return idx_bootstrap_samples, inbag_samples

def node_count_forest(forest):
    n_trees = forest.n_estimators
    node_count_by_tree = np.zeros((n_trees), dtype=np.uint32)
    
    for i in range(n_trees):
        node_count_by_tree[i] = forest.estimators_[i].tree_.__getstate__()['node_count']
        
    return np.amax(node_count_by_tree)


# -----------------------------------------------------------------------------
# FIRST CODE
# -----------------------------------------------------------------------------

@nb.njit("float64(float64[:], int64[:,:], uint32[:,:], int64[:], float64)", nogil=True, cache=False, parallel=False)
def conditional_CDF_1(output_samples, samples_nodes, idx_bootstrap_samples, X_nodes, y):
    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    
    CDF_tree = np.zeros((n_trees), dtype=np.float64)
    for i in range(n_trees):
        count = 0.
        for j in range(n_samples):
            if samples_nodes[idx_bootstrap_samples[j,i],i] == X_nodes[i]:
                if output_samples[idx_bootstrap_samples[j,i]] <= y:
                    CDF_tree[i] += 1
                count += 1
        CDF_tree[i] /= count
    
    return CDF_tree.mean()

@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:])", nogil=True, cache=False, parallel=True)
def _compute_conditional_quantile_with_bootstrap_data_1(output_samples, samples_nodes, idx_bootstrap_samples, X_nodes, alpha):
    n_samples = output_samples.shape[0]
    n_quantiles = X_nodes.shape[0]
    n_alphas = alpha.shape[0]
    quantiles = np.zeros((n_quantiles, n_alphas), dtype=np.float64)
    
    output_samples_sorted = np.sort(output_samples)
    for i in nb.prange(n_quantiles):
        for j, alpha_temp in enumerate(alpha):       
            k = 0
            ans = 0.
            while(k < n_samples and ans < alpha_temp):
                ans = conditional_CDF_1(output_samples, samples_nodes, idx_bootstrap_samples, X_nodes[i,:], output_samples_sorted[k])
                k += 1
            quantiles[i,j] = output_samples_sorted[k-1]

    return quantiles

@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:])", nogil=True, cache=False, parallel=True)
def _compute_conditional_quantile_with_bootstrap_data_2(output_samples, samples_nodes, idx_bootstrap_samples, X_nodes, alpha):
    n_samples = output_samples.shape[0]
    n_quantiles = X_nodes.shape[0]
    n_alphas = alpha.shape[0]
    quantiles = np.zeros((n_quantiles, n_alphas), dtype=np.float64)
    
    output_samples_sorted = np.sort(output_samples)
    for i in nb.prange(n_quantiles):
        k = 0
        alpha_temp = alpha[k]
        for j in range(n_samples):
            ans = conditional_CDF_1(output_samples, samples_nodes, idx_bootstrap_samples, X_nodes[i,:], output_samples_sorted[j])
            if ans > alpha_temp:
                quantiles[i, k] = output_samples_sorted[j]
                k += 1
                if k != n_alphas:
                    alpha_temp = alpha[k]
                else:
                    break

    return quantiles


# -----------------------------------------------------------------------------
# SECOND CODE
# -----------------------------------------------------------------------------

@nb.njit("float64[:](float64[:], float64[:], int64[:,:], uint32[:,:], int64[:])", nogil=True, cache=False, parallel=False)
def conditional_CDF_2(output_samples, output_samples_sorted, samples_nodes, idx_bootstrap_samples, X_nodes):
    n_samples = samples_nodes.shape[0]
    n_trees = samples_nodes.shape[1]
    
    CDF_by_tree = np.zeros((n_samples, n_trees), dtype=np.float64)
    CDF_forest = np.empty((n_samples), dtype=np.float64)
    
    for i in range(n_trees):
        count = 0.
        for j in range(n_samples):
            if samples_nodes[idx_bootstrap_samples[j,i],i] == X_nodes[i]:
                count += 1
                k = n_samples - 1
                while(0 <= k and output_samples[idx_bootstrap_samples[j,i]] <= output_samples_sorted[k]):
                    CDF_by_tree[k,i] += 1
                    k -= 1
        for l in range(n_samples):
            CDF_by_tree[l,i] /= count
    
    for i in range(n_samples):
        res = 0.
        for j in range(n_trees):
            res += CDF_by_tree[i,j]
        CDF_forest[i] = res/n_trees

    return CDF_forest

@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], int64[:,:], float64[:])", nogil=True, cache=False, parallel=True)
def _compute_conditional_quantile_with_bootstrap_data_3(output_samples, samples_nodes, idx_bootstrap_samples, X_nodes, alpha):
    n_quantiles = X_nodes.shape[0]
    n_alphas = alpha.shape[0]
    quantiles = np.zeros((n_quantiles, n_alphas), dtype=np.float64)
    
    output_samples_sorted = np.sort(output_samples)
    for i in nb.prange(n_quantiles):
        CDF_forest = conditional_CDF_2(output_samples,
                                       output_samples_sorted,
                                       samples_nodes,
                                       idx_bootstrap_samples,
                                       X_nodes[i,:])
        
        quantiles[i, :] = np.array([
                            output_samples_sorted[
                                    np.argmax((CDF_forest >= alpha_var).astype(np.uint32))] 
                            for alpha_var in alpha])

    return quantiles


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
# -----------------------------------------------------------------------------
    
np.random.seed(0)

n_samples = 10**3
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

idx_bootstrap_samples, inbag_samples = compute_inbag_samples(n_samples, quantForest)

start = time.time()
a = _compute_conditional_quantile_with_bootstrap_data_1(output_samples, samples_nodes, idx_bootstrap_samples, X_nodes, alpha)
print(time.time() - start)
    
start = time.time()
b = _compute_conditional_quantile_with_bootstrap_data_2(output_samples, samples_nodes, idx_bootstrap_samples, X_nodes, alpha)
print(time.time() - start)

start = time.time()
c = _compute_conditional_quantile_with_bootstrap_data_3(output_samples, samples_nodes, idx_bootstrap_samples, X_nodes, alpha)
print(time.time() - start)

start = time.time()
idx_bootstrap_samples, inbag_samples = compute_inbag_samples(n_samples, quantForest)
n_leaf = np.int64(node_count_forest(quantForest))
d = _compute_conditional_quantile_with_bootstrap_data_4(n_leaf, output_samples, samples_nodes, inbag_samples, X_nodes, alpha)
print(time.time() - start)

start = time.time()
e = quantForest.predict(X_nodes_val, alpha, used_bootstrap_samples=False)
print(time.time() - start)

start = time.time()
f = quantForest.predict(X_nodes_val, alpha, used_bootstrap_samples=True)
print(time.time() - start)

print((a != b ).sum())
print((b != c ).sum())
print((a != c ).sum())
print((b != d ).sum())
print((c != d ).sum())
print((e != c ).sum())
print((c != f ).sum())
print((e != f ).sum())
print((e != d ).sum())
print((d != f ).sum())
