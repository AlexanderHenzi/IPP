# -*- coding: utf-8 -*-

import math
import numba as nb
import numpy as np
import scipy.stats as ss
import seaborn as sns

from sklearn.model_selection import KFold

from qosa import QuantileRegressionForest
from qosa import cross_validate


def golden_search_algorithm(F, a, b, tol=2, Niter=10**3):
    assert (a < b) , "Error , we must have a < b"
    
    # Initialization
    tau = (math.sqrt(5) - 1)*0.5
    an, bn, x1, x2 = a, b, round(tau*a + (1 - tau)*b), round((1 - tau)*a + tau*b)
    Fx1, Fx2 = F(x1), F(x2)
       
    # Iteration of the Golden Search algorithm
    i = 0
    while(i < Niter and (bn - an) > tol):
        if (Fx1 <= Fx2):
            an, bn, x1, x2 = an, x2, round(tau*an + (1 - tau)*x2), x1
            Fx1, Fx2 = F(x1), Fx1
        else:
            an, bn, x1, x2 = x1, bn, x2, round((1 - tau)*x1 + tau*bn)
            Fx1, Fx2 = Fx2, F(x2)

        i += 1
            
    return round((an + bn)*0.5), i


# -----------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# -----------------------------------------------------------------------------
#
# Part to test the golden search algorithm on the cross validate function
#    
# -----------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# -----------------------------------------------------------------------------

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


def check_function_error(estimator, X_test, y_test, alpha, used_bootstrap_samples):
    cond_quant = estimator.predict(X_test, 
                                   alpha,
                                   used_bootstrap_samples=used_bootstrap_samples)
    u = y_test.reshape(-1,1) - cond_quant
    return _averaged_check_function_alpha_array(u, alpha)


# ----------------------------------------------------------------------------------
# Compute of an empirical way the numerator of the qosa index function of the leaves
# ----------------------------------------------------------------------------------
    
n_samples = 10**5
X = np.random.exponential(size = (n_samples,2))
y = X[:,0] - X[:,1]
X = X[:,0].reshape(-1,1)

n_estimators = 3*10**2
shuffle = True
alpha = np.array([0.1, 0.3, 0.7, 0.99])
n_alphas = alpha.shape[0]
scoring = check_function_error

min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(1000), num=120, endpoint=True, dtype=int))
n_leaf = min_samples_leaf.shape[0]

n_fold = [5, 10]
length_n_fold = len(n_fold)
cross_val_mean_values = np.zeros((length_n_fold, n_leaf, n_alphas), dtype=np.float64)
optimum_leaf_by_alpha = np.zeros((length_n_fold, n_alphas), dtype=np.float64)
for i, n_splits in enumerate(n_fold):

    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=0)
    scoring_params = {'alpha':alpha, 'used_bootstrap_samples':False}
    for j in range(n_leaf):
        print('nfold: ', n_splits, j)
        cross_val_mean_values[i,j,:] = cross_validate(
                                    estimator=QuantileRegressionForest(
                                                        n_estimators=n_estimators,
                                                        min_samples_split=min_samples_leaf[j]*2, 
                                                        min_samples_leaf=min_samples_leaf[j],
                                                        random_state=0), 
                                    X=X,
                                    y=y,
                                    cv=cv,
                                    scoring=check_function_error,
                                    scoring_params=scoring_params).mean(axis=0)
                                    
    # -----------------------------------------------------------------------------
    # Part to find the exact value of leaf with the previous algorithm
    # -----------------------------------------------------------------------------

    def generate_estimator(estimator_class, **kwargs):
        return lambda x: estimator_class(min_samples_split=x*2, 
                                         min_samples_leaf=x,
                                         **kwargs)

    def generate_function_to_optimize(estimator, **kwargs):
        return lambda x: cross_validate(estimator=estimator(x), **kwargs).mean()
    
    estimator = generate_estimator(QuantileRegressionForest, n_estimators=n_estimators, random_state=0)
    
    for j in range(n_alphas):
        scoring_params = {'alpha':np.array([alpha[j]]), 'used_bootstrap_samples':False}
        function_to_optimize = generate_function_to_optimize(estimator,
                                                             X=X,
                                                             y=y,
                                                             cv=cv,
                                                             scoring=check_function_error,
                                                             scoring_params=scoring_params)

        temp = golden_search_algorithm(function_to_optimize, 5, 1000, tol=2)
        optimum_leaf_by_alpha[i, j] = temp[0]
        print('alpha = ', alpha[j], ' in ', temp[1], ' iterations')

# -----------------------------------------------------------------------------
# Compute the numerator of the qosa index
# -----------------------------------------------------------------------------

def numerator_qosa_index_given_1(alpha):
    q_X2 = ss.expon.ppf(1 - alpha)
    t_EY_X1 = alpha - np.exp(-q_X2)*(1 + q_X2)
    return -t_EY_X1 

true_numerator = numerator_qosa_index_given_1(alpha)

# -----------------------------------------------------------------------------
# Plot the results
# -----------------------------------------------------------------------------    

import matplotlib.pyplot as plt
plt.switch_backend('Agg') # very important to plot on cluster
from matplotlib.backends.backend_pdf import PdfPages # to make several figures in one pdf
from qosa.plots import set_style_paper

colors = sns.color_palette('bright')
set_style_paper()
# The rcParams 'xtick.bottom' and 'ytick.left' can be used to set the ticks on or off
# because the seaborn "white" style deactivate the tick marks
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

pdf_pages = PdfPages('test_optimization_algorithm_model_diff_expo.pdf')

for i in range(n_alphas):
    fig, axes = plt.subplots(figsize=(12, 8))
    for j in range(length_n_fold):
        if j == 0:
            axes.axhline(true_numerator[i], lw=2, linestyle='--', label='True value', color=colors[length_n_fold])
        axes.plot(min_samples_leaf, cross_val_mean_values[j,:,i], lw=2, label=r'$N_{fold} = %d $' %(n_fold[j],), color=colors[j])
        axes.axvline(optimum_leaf_by_alpha[j, i], lw=1, linestyle='-.', label='optimal leaf \n with algorithm', color=colors[j])
        axes.set_xlabel('Values of min_samples_leaf', fontsize=14)
        axes.tick_params(axis = 'both', labelsize = 14)
        axes.legend(loc='best', frameon=True, fontsize = 14)
        fig.suptitle(r'$ \mathbb{E} \left[ \psi \left( Y, q_{\alpha = %.2f} (Y|X_1) \right) \right]$'
                     ' with $N_{sample}=%d, N_{tree}=%d, N_{min\_samples\_leaf}=%d$' 
                        % (alpha[i], n_samples, n_estimators, min_samples_leaf.shape[0]), 
                        fontsize = 14, y=1.02)
    fig.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

pdf_pages.close()
