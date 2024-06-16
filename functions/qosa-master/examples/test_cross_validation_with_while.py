import numba as nb
import numpy as np
import time
from time import perf_counter
from sklearn.model_selection import KFold
from qosa import QuantileRegressionForest

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

from qosa import cross_validate

n_samples = 4*10**4
X = np.random.exponential(size = (n_samples,2))
y = X[:,0] - X[:,1]
X = X[:,0].reshape(-1,1)

alpha = np.array([0.1, 0.3, 0.7, 0.9])
min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(2000), 61, endpoint=True, dtype=int))
used_bootstrap_samples = False
shuffle = True
n_estimators = 2*10**2
n_splits = 4
n_alphas = alpha.shape[0]

def check_function_error(estimator, X, y):
    cond_quant = estimator.predict(X, 
                                   alpha,
                                   used_bootstrap_samples=used_bootstrap_samples)
    u = y.reshape(-1,1) - cond_quant
    return _averaged_check_function_alpha_array(u, alpha)

n_leaf = min_samples_leaf.shape[0]

cross_val_values = np.zeros((n_leaf, n_splits, n_alphas), dtype=np.float64)
cross_val_mean_values = np.zeros((n_leaf, n_alphas), dtype=np.float64)

random_state = np.arange(min_samples_leaf.shape[0])

start = time.time()
for i in range(n_leaf):
    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state[2])
    cross_val_values[i,:,:] = cross_validate(
                                estimator=QuantileRegressionForest(
                                                    n_estimators=n_estimators,
                                                    min_samples_split=min_samples_leaf[i]*2, 
                                                    min_samples_leaf=min_samples_leaf[i],
                                                    random_state=random_state[2]), 
                                X=X,
                                y=y,
                                scoring=check_function_error,
                                cv=cv)['test_score']
    
    cross_val_mean_values[i,:] = cross_val_values[i,:,:].mean(axis=0)
print("Time for the first method: ", time.time() - start)


# -----------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# -----------------------------------------------------------------------------
#
# Part to test the curve fitting
#
# -----------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
plt.switch_backend('Agg') # very important to plot on cluster

#from scipy.optimize import curve_fit
#
#def func(x, b, c, d, e):
#    return b*x**3 + c*x**2 + d*x + e
#
#x_values = min_samples_leaf
#y_values = cross_val_mean_values[:,0]
#popt, pcov = curve_fit(func, x_values, y_values)
#
#x_estimate = np.arange(5, 1500, 1)
#y_estimate = func(x_estimate, *popt)

from qosa.plots import set_style_paper

set_style_paper()
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8), sharex=True)
k = 0
for i in range(2):
    for j in range(2):
        axes[i,j].plot(min_samples_leaf, cross_val_mean_values[:,k], lw=2, label=r'$\alpha = %.2f $' %(alpha[k],))
        axes[i,j].legend(loc='best', frameon=True, fontsize = 14)
        if i==1:
            axes[i,j].set_xlabel('Values of min_samples_leaf', fontsize=14)
        axes[i,j].tick_params(axis = 'both', labelsize = 14)
        k += 1

fig.suptitle(r'$ \mathbb{E} \left[ \psi \left( Y, q_\alpha (Y|X_1) \right) \right]$'
             ' with $N_{sample}=%d, N_{tree}=%d, N_{min\_samples\_leaf}=%d, N_{fold}=%d$' 
                % (n_samples, n_estimators, min_samples_leaf.shape[0], n_splits), fontsize = 16, y=1.02)
fig.tight_layout()
fig.savefig('numerator_qosa_index.pdf', bbox_inches='tight')

# -----------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# -----------------------------------------------------------------------------
#
# Second version of the code with the while 
#
# -----------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# -----------------------------------------------------------------------------

#n_leaf_optim = np.zeros((n_alphas), dtype=np.uint32)
#idx_alpha = np.arange(n_alphas)
#
#cross_val_mean_values_bis = np.zeros((n_leaf, n_alphas), dtype=np.float64)
#
#start = time.time()
#cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state[0])
#ans = cross_validate(
#                    estimator=QuantileRegressionForest(
#                                        n_estimators=n_estimators,
#                                        min_samples_split=min_samples_leaf[0]*2, 
#                                        min_samples_leaf=min_samples_leaf[0],
#                                        random_state=random_state[0]), 
#                    X=X,
#                    y=y,
#                    scoring=check_function_error,
#                    cv=cv)['test_score'].mean(axis=0)
#
#cross_val_mean_values_bis[0, :] = ans
#
#i = 1
#while(i < n_leaf):
#    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state[0])
#    temp = cross_validate(
#                        estimator=QuantileRegressionForest(
#                                            n_estimators=n_estimators,
#                                            min_samples_split=min_samples_leaf[i]*2, 
#                                            min_samples_leaf=min_samples_leaf[i],
#                                            random_state=random_state[0]), 
#                        X=X,
#                        y=y,
#                        scoring=check_function_error,
#                        cv=cv)['test_score'].mean(axis=0)
#    
#    cross_val_mean_values_bis[i, idx_alpha] = temp
#    
#    del_idx = []
#    for j in range(n_alphas):
#        if temp[j] > ans[j]:
#            n_leaf_optim[idx_alpha[j]] = min_samples_leaf[i-1]
#            del_idx.append(j)
#    
#    if del_idx:
#        idx_alpha = np.delete(idx_alpha, del_idx)
#        alpha = np.delete(alpha, del_idx)
#        ans = np.delete(temp, del_idx)
#        n_alphas = alpha.size
#    else:
#        ans = temp
#    
#    if n_alphas != 0:
#        i += 1
#    else:
#        break
#    
#print("Time for the second method : ", time.time() - start)
#
#
#n_optim_leaf = np.argmin(cross_val_mean_values, axis=0)
#print("Optimal leaf for the first: ", min_samples_leaf[n_optim_leaf],"\n")
#print("Optimal leaf for the second: ", n_leaf_optim,"\n")
#print(cross_val_mean_values,"\n")
#print(cross_val_mean_values_bis)
