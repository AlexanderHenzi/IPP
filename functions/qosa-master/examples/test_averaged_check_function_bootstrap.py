# -*- coding: utf-8 -*-

import numba as nb
import numpy as np
import time

# from sklearn.ensemble.forest import _generate_sample_indices

# def compute_inbag_idx(n_samples, forest):
#     n_trees = forest.n_estimators
#     samples_idx = np.empty((n_samples, n_trees), dtype=np.uint32, order='F')
#     for t_idx in range(n_trees):
#         samples_idx[:, t_idx] = _generate_sample_indices(forest.estimators_[t_idx].random_state, 
#                                                          n_samples,
#                                                          n_samples)
#     return samples_idx

@nb.njit("float64(float64[:], float64)", nogil=True, cache=False, parallel=False)
def _averaged_check_function_alpha_float_unparallel(u, alpha):
    """
    Definition of the check function also called pinball loss function.
    """
    
    n_samples = u.shape[0]
    res = 0.    
    for i in range(n_samples):
        res += u[i]*(alpha - (u[i] < 0.))
    
    res = res/n_samples
    return res

# @nb.njit("float64[:,:](float64[:], int64[:,:], float64[:])", nogil=True, cache=False, parallel=True)
# def _compute_conditional_part_qosa_index(output_samples, samples_nodes, alpha):
#     """
#     Compute the expected optimal value of the conditional contrast function
#     """
    
#     # Number of trees in the forest 
#     n_tree = samples_nodes.shape[1]
    
#     # Size of the sample
#     n_samples = samples_nodes.shape[0]
    
#     # Number of conditional parts to compute
#     n_alphas = alpha.size  # Number of probabilities
    
#     # Unique leaf nodes of each tree
#     samples_nodes = np.asfortranarray(samples_nodes)
#     index_leaf_nodes_by_tree = [np.unique(samples_nodes[:,i]) for i in range(n_tree)]
    
#     expectation_forest_by_alpha = np.empty((n_alphas, 2), dtype=np.float64) # 2 for the classical and weighted mean
#     expectation_by_tree = np.empty((n_alphas, n_tree, 2), dtype=np.float64) # 2 for the classical and weighted mean
#     for i in nb.prange(n_tree):
#     #for i,index_leaf_nodes in enumerate(index_leaf_nodes_by_tree):
#         index_leaf_nodes = index_leaf_nodes_by_tree[i]
#         n_leaf_nodes_for_the_tree = index_leaf_nodes.shape[0]
#         expectation_by_node = np.empty((n_alphas, n_leaf_nodes_for_the_tree), dtype=np.float64)
#         node_weights = np.empty((n_leaf_nodes_for_the_tree), dtype=np.float64)
#         Y_leaf_node_temp = np.empty((n_samples), dtype=np.float64)
        
#         for j, index_leaf_nodes_temp in enumerate(index_leaf_nodes):
#             n_samples_Y_leaf_node = 0    
#             for k in range(n_samples):
#                 if samples_nodes[k,i] == index_leaf_nodes_temp:
#                     Y_leaf_node_temp[n_samples_Y_leaf_node] = output_samples[k]
#                     n_samples_Y_leaf_node += 1
                    
#             Y_leaf_node = Y_leaf_node_temp[:np.int64(n_samples_Y_leaf_node)]
#             argsort_Y_leaf_node = np.argsort(Y_leaf_node)
#             node_weights[j] = Y_leaf_node.shape[0]/n_samples
#             for l,alpha_temp in enumerate(alpha):
#                 ans = _averaged_check_function_alpha_float_unparallel(
#                                 Y_leaf_node - Y_leaf_node[argsort_Y_leaf_node[0]],
#                                 alpha_temp) 
#                 m=1
#                 while(m < Y_leaf_node.shape[0]):
#                     temp = _averaged_check_function_alpha_float_unparallel(
#                                 Y_leaf_node - Y_leaf_node[argsort_Y_leaf_node[m]],
#                                 alpha_temp)
#                     if(temp<ans):
#                         ans = temp
#                         m+=1
#                     else:
#                         break
#                 expectation_by_node[l,j] = ans
#         for n in range(n_alphas):
#             expectation_by_tree[n,i,0] = expectation_by_node[n,:].sum()/n_leaf_nodes_for_the_tree
#             expectation_by_tree[n,i,1] = (expectation_by_node[n,:]*node_weights).sum()

#     for i in nb.prange(n_alphas):      
#         expectation_forest_by_alpha[i,0] = expectation_by_tree[i,:,0].sum()/n_tree # First column: classical mean
#         expectation_forest_by_alpha[i,1] = expectation_by_tree[i,:,1].sum()/n_tree # Second column: weighted mean
    
#     return expectation_forest_by_alpha


# @nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], float64[:])", nogil=True, cache=False, parallel=True)
# def _compute_conditional_part_qosa_index_bootstrap(output_samples, samples_nodes, idx_bootstrap_samples, alpha):
#     """
#     Compute the expected optimal value of the conditional contrast function
#     """
    
#     # Number of trees in the forest 
#     n_tree = samples_nodes.shape[1]
    
#     # Size of the sample
#     n_samples = samples_nodes.shape[0]
    
#     # Number of conditional parts to compute
#     n_alphas = alpha.size  # Number of probabilities
    
#     # Unique leaf nodes of each tree
#     samples_nodes = np.asfortranarray(samples_nodes)
#     index_leaf_nodes_by_tree = [np.unique(samples_nodes[:,i]) for i in range(n_tree)]
    
#     expectation_forest_by_alpha = np.empty((n_alphas, 2), dtype=np.float64) # 2 for the classical and weighted mean
#     expectation_by_tree = np.empty((n_alphas, n_tree, 2), dtype=np.float64) # 2 for the classical and weighted mean
#     for i in nb.prange(n_tree):
#     #for i,index_leaf_nodes in enumerate(index_leaf_nodes_by_tree):
#         index_leaf_nodes = index_leaf_nodes_by_tree[i]
#         n_leaf_nodes_for_the_tree = index_leaf_nodes.shape[0]
#         expectation_by_node = np.empty((n_alphas, n_leaf_nodes_for_the_tree), dtype=np.float64)
#         node_weights = np.empty((n_leaf_nodes_for_the_tree), dtype=np.float64)
#         Y_leaf_node_temp = np.empty((n_samples), dtype=np.float64)
        
#         for j, index_leaf_nodes_temp in enumerate(index_leaf_nodes):
#             n_samples_Y_leaf_node = 0    
#             for k in range(n_samples):
#                 if samples_nodes[idx_bootstrap_samples[k,i],i] == index_leaf_nodes_temp:
#                     Y_leaf_node_temp[n_samples_Y_leaf_node] = output_samples[idx_bootstrap_samples[k,i]]
#                     n_samples_Y_leaf_node += 1
                    
#             Y_leaf_node = Y_leaf_node_temp[:np.int64(n_samples_Y_leaf_node)]
#             argsort_Y_leaf_node = np.argsort(Y_leaf_node)
#             node_weights[j] = Y_leaf_node.shape[0]/n_samples
#             for l,alpha_temp in enumerate(alpha):
#                 ans = _averaged_check_function_alpha_float_unparallel(
#                                 Y_leaf_node - Y_leaf_node[argsort_Y_leaf_node[0]],
#                                 alpha_temp) 
#                 m=1
#                 while(m < Y_leaf_node.shape[0]):
#                     temp = _averaged_check_function_alpha_float_unparallel(
#                                 Y_leaf_node - Y_leaf_node[argsort_Y_leaf_node[m]],
#                                 alpha_temp)
#                     if(temp<=ans):
#                         ans = temp
#                         m+=1
#                     else:
#                         break
#                 expectation_by_node[l,j] = ans
#         for n in range(n_alphas):
#             expectation_by_tree[n,i,0] = expectation_by_node[n,:].sum()/n_leaf_nodes_for_the_tree
#             expectation_by_tree[n,i,1] = (expectation_by_node[n,:]*node_weights).sum()

#     for i in nb.prange(n_alphas):      
#         expectation_forest_by_alpha[i,0] = expectation_by_tree[i,:,0].sum()/n_tree # First column: classical mean
#         expectation_forest_by_alpha[i,1] = expectation_by_tree[i,:,1].sum()/n_tree # Second column: weighted mean
    
#     return expectation_forest_by_alpha


@nb.njit("float64[:,:](float64[:], int64[:,:], uint32[:,:], float64[:], boolean)", nogil=True, cache=False, parallel=True)
def _compute_expected_value_check_function(output_samples, samples_nodes, idx_bootstrap_samples, alpha, used_bootstrap_samples):
    """
    Compute the expected optimal value of the averaged check function.
    """
    
    # Number of trees in the forest 
    n_trees = samples_nodes.shape[1]
    
    # Size of the sample
    n_samples = samples_nodes.shape[0]
    
    # Number of conditional parts to compute
    n_alphas = alpha.size  # Number of probabilities
    
    # Unique leaf nodes of each tree
    samples_nodes = np.asfortranarray(samples_nodes)
    idx_leaves_by_tree = [np.unique(samples_nodes[:,i]) for i in range(n_trees)]
    
    expectation_forest_by_alpha = np.empty((n_alphas, 2), dtype=np.float64) # 2 for the classical and weighted mean
    expectation_by_tree = np.empty((n_alphas, n_trees, 2), dtype=np.float64) # 2 for the classical and weighted mean
    for i in nb.prange(n_trees):
    #for i,idx_leaves_for_tree_i in enumerate(idx_leaves_by_tree):
        idx_leaves_for_tree_i = idx_leaves_by_tree[i]
        n_idx_leaves_for_tree_i = idx_leaves_for_tree_i.shape[0]
        expectation_by_node = np.empty((n_alphas, n_idx_leaves_for_tree_i), dtype=np.float64)
        node_weights = np.empty((n_idx_leaves_for_tree_i), dtype=np.float64)
        Y_leaves = np.empty((n_samples), dtype=np.float64)
        
        for j, idx_current_leaf in enumerate(idx_leaves_for_tree_i):
            n_Y_leaves = 0
            
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
            
            # np.int64(n_Y_leaves): np.int64 necessary because of a bug of numba when using parallel=True
            Y_leaves_node = Y_leaves[:np.int64(n_Y_leaves)]
            argsort_Y_leaves_node = np.argsort(Y_leaves_node)
            node_weights[j] = Y_leaves_node.shape[0]/n_samples
            m=0
            for l, alpha_temp in enumerate(alpha):
                if m < Y_leaves_node.shape[0]:
                    ans = _averaged_check_function_alpha_float_unparallel(
                                    Y_leaves_node - Y_leaves_node[argsort_Y_leaves_node[m]],
                                    alpha_temp)
                else:
                    for o in range(l, n_alphas):
                        expectation_by_node[o,j] = _averaged_check_function_alpha_float_unparallel(
                                                        Y_leaves_node - Y_leaves_node[argsort_Y_leaves_node[-1]],
                                                        alpha[o])
                    break
                
                if l == 0:
                    m=1
                else:
                    m+=1
                while(m < Y_leaves_node.shape[0]):
                    temp = _averaged_check_function_alpha_float_unparallel(
                                Y_leaves_node - Y_leaves_node[argsort_Y_leaves_node[m]],
                                alpha_temp)
                    if(temp <= ans):
                        ans = temp
                        m+=1
                    else:
                        m-=1
                        break
                expectation_by_node[l,j] = ans
        for n in range(n_alphas):
            expectation_by_tree[n, i, 0] = expectation_by_node[n, :].sum()/n_idx_leaves_for_tree_i
            expectation_by_tree[n, i, 1] = (expectation_by_node[n, :]*node_weights).sum()

    for i in nb.prange(n_alphas):      
        expectation_forest_by_alpha[i, 0] = expectation_by_tree[i, :, 0].sum()/n_trees # First column: classical mean
        expectation_forest_by_alpha[i, 1] = expectation_by_tree[i, :, 1].sum()/n_trees # Second column: weighted mean
    
    return expectation_forest_by_alpha

# -----------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# -----------------------------------------------------------------------------
    
from qosa import ExpectedCheckFunctionLeaves

n_sample = 10**4
X = np.random.exponential(size = (n_sample,2))
y = X[:,0] - X[:,1]
X1 = X[:,0].reshape(-1,1)

alpha = np.array([0.9, 0.99, 0.995])
min_samples_leaf = 10

estimator = ExpectedCheckFunctionLeaves(
                    n_estimators=10**2,
                    min_samples_split=min_samples_leaf*2, 
                    min_samples_leaf=min_samples_leaf,
                    n_jobs=-1)
estimator.fit(X1, y)
samples_nodes = estimator._samples_nodes

# a = _compute_conditional_part_qosa_index(y, samples_nodes, alpha)

idx_bootstrap_samples, _ = estimator._compute_inbag_samples()
# b = _compute_conditional_part_qosa_index_bootstrap(y, samples_nodes, idx_bootstrap_samples, alpha)

start = time.time()
for i in range(10):
    used_bootstrap_samples = False
    c = _compute_expected_value_check_function(y, samples_nodes, idx_bootstrap_samples, alpha, used_bootstrap_samples)
print('Elapsed time: ', time.time() - start)

start = time.time()
for i in range(10):
    used_bootstrap_samples = True
    d = _compute_expected_value_check_function(y, samples_nodes, idx_bootstrap_samples, alpha, used_bootstrap_samples)
print('Elapsed time: ', time.time() - start)

# print( (a != c).sum())
# print( (b != d).sum())

start = time.time()
for i in range(10):
    e = estimator.predict(alpha, used_bootstrap_samples=False)
print('Elapsed time: ', time.time() - start)

start = time.time()
for i in range(10):
    f = estimator.predict(alpha, used_bootstrap_samples=True)
print('Elapsed time: ', time.time() - start)

print((c != e).sum())
print((d != f).sum())


