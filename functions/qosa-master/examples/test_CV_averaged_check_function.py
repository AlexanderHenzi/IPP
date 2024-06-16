# -*- coding: utf-8 -*-

import numba as nb 
import numpy as np


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

@nb.njit("float64[:,:](float64[:], int64[:,:], float64[:])", nogil=True, cache=False, parallel=True)
def _compute_conditional_part_qosa_index(output_sample, samples_nodes, alpha):
    """
    Compute the expected optimal value of the conditional contrast function
    """
    
    # Number of trees in the forest 
    n_tree = samples_nodes.shape[1]
    
    # Size of the sample
    n_samples = samples_nodes.shape[0]
    
    # Number of conditional parts to compute
    n_alphas = alpha.size  # Number of probabilities
    
    # Unique leaf nodes of each tree
    samples_nodes = np.asfortranarray(samples_nodes)
    index_leaf_nodes_by_tree = [np.unique(samples_nodes[:,i]) for i in range(n_tree)]
    
    expectation_forest_by_alpha = np.empty((n_alphas, 2), dtype=np.float64) # 2 for the classical and weighted mean
    expectation_by_tree = np.empty((n_alphas, n_tree, 2), dtype=np.float64) # 2 for the classical and weighted mean
    for i in nb.prange(n_tree):
    #for i,index_leaf_nodes in enumerate(index_leaf_nodes_by_tree):
        index_leaf_nodes = index_leaf_nodes_by_tree[i]
        n_leaf_nodes_for_the_tree = index_leaf_nodes.shape[0]
        expectation_by_node = np.empty((n_alphas, n_leaf_nodes_for_the_tree), dtype=np.float64)
        node_weights = np.empty((n_leaf_nodes_for_the_tree), dtype=np.float64)
        Y_leaf_node_temp = np.empty((n_samples), dtype=np.float64)
        
        for j, index_leaf_nodes_temp in enumerate(index_leaf_nodes):
            n_samples_Y_leaf_node = 0    
            for k in range(n_samples):
                if samples_nodes[k,i] == index_leaf_nodes_temp:
                    Y_leaf_node_temp[n_samples_Y_leaf_node] = output_sample[k]
                    n_samples_Y_leaf_node += 1
                    
            Y_leaf_node = Y_leaf_node_temp[:n_samples_Y_leaf_node]
            argsort_Y_leaf_node = np.argsort(Y_leaf_node)
            node_weights[j] = Y_leaf_node.shape[0]/n_samples
            for l,alpha_temp in enumerate(alpha):
                ans = _averaged_check_function_alpha_float_unparallel(
                                Y_leaf_node - Y_leaf_node[argsort_Y_leaf_node[0]],
                                alpha_temp) 
                m=1
                while(m < Y_leaf_node.shape[0]):
                    temp = _averaged_check_function_alpha_float_unparallel(
                                Y_leaf_node - Y_leaf_node[argsort_Y_leaf_node[m]],
                                alpha_temp)
                    if(temp<ans):
                        ans = temp
                        m+=1
                    else:
                        break
                expectation_by_node[l,j] = ans
        for n in range(n_alphas):
            expectation_by_tree[n,i,0] = expectation_by_node[n,:].sum()/n_leaf_nodes_for_the_tree
            expectation_by_tree[n,i,1] = (expectation_by_node[n,:]*node_weights).sum()

    for i in nb.prange(n_alphas):      
        expectation_forest_by_alpha[i,0] = expectation_by_tree[i,:,0].sum()/n_tree # First column: classical mean
        expectation_forest_by_alpha[i,1] = expectation_by_tree[i,:,1].sum()/n_tree # Second column: weighted mean
    
    return expectation_forest_by_alpha


# -----------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# -----------------------------------------------------------------------------
    
from sklearn.ensemble import RandomForestRegressor

n_sample = 10**4
X = np.random.exponential(size = (n_sample,2))
y = X[:,0] - X[:,1]
X1 = X[:,0].reshape(-1,1)

alpha = np.array([0.5, 0.7, 0.99])
min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(1500), 10, endpoint=True, dtype=int))
averaged = np.zeros((min_samples_leaf.shape[0], alpha.shape[0]))

for i in range(min_samples_leaf.shape[0]):
    print(i)
    estimator = RandomForestRegressor(
                        n_estimators=200,
                        min_samples_split=min_samples_leaf[i]*2, 
                        min_samples_leaf=min_samples_leaf[i],
                        n_jobs=-1)
    estimator.fit(X1, y)
    samples_nodes = estimator.apply(X1)
    averaged[i,:] = _compute_conditional_part_qosa_index(y, samples_nodes, alpha)[:,0]

print(averaged)


