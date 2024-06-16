# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
plt.switch_backend('Agg') 
import numba as nb
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

from qosa import cross_validation_forest, QuantileRegressionForest


# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------
#
# Functions needed to compute the QOSA indices
#
# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------

@nb.njit("float64(float64[:], float64)", nogil=True, cache=False, parallel=True)
def _averaged_check_function_alpha_float_parallel(u, alpha):
    """
    Definition of the check function also called pinball loss function.
    """

    n_samples = u.shape[0]
    res = 0.    
    for i in nb.prange(n_samples):
        res += u[i]*(alpha - (u[i] < 0.))
    
    res /= n_samples
    return res


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


def _qosa_forest_compute_quantile(X1,
                                  X2, 
                                  Y1,
                                  Y2,
                                  dim,
                                  alpha,
                                  method,
                                  n_estimators,
                                  min_samples_leaf,
                                  used_bootstrap_samples,
                                  optim_by_CV,
                                  CV_strategy,
                                  n_fold,
                                  n_boot):
    """
    Compute the qosa indices with the Random Forest method by plugging the
    conditional quantiles
    """
    
    n_alphas = alpha.shape[0]
    n_samples = Y1.shape[0]
    
    numerator = np.empty((dim, n_alphas), dtype=np.float64)
    min_samples_leaf_by_dim_and_alpha = np.empty((n_alphas, dim), dtype=np.uint32) if optim_by_CV else None
    qosa_indices = np.empty((dim, n_alphas, n_boot), dtype=np.float64)
    
    for i in range(n_boot):
        print(i)
        # Bootstrap sample indexes
        # The first iteration is computed over the all sample.
        if i > 0:
            boot_idx = np.random.randint(0, n_samples, size=(n_samples, ))
        else:
            boot_idx = range(n_samples)
        
        # Compute the denominator of the indices
        alpha_quantile = np.percentile(Y1[boot_idx], q=alpha*100)
        denominator = _averaged_check_function_alpha_array(
                                                    Y1[boot_idx].reshape(-1,1) - alpha_quantile,
                                                    alpha)
    
       # Compute the numerator of the indices for each variable
        for j in range(dim):
            X1_j = X1[boot_idx, j]
            X2_j = X2[boot_idx, j]
    
            if i == 0 :
                _, min_samples_leaf_by_dim_and_alpha[:, j] = cross_validation_forest(
                                                X=X2_j,
                                                y=Y2,
                                                alpha=alpha,
                                                min_samples_leaf=min_samples_leaf,
                                                method=method,
                                                n_estimators=n_estimators,
                                                used_bootstrap_samples=used_bootstrap_samples,
                                                CV_strategy=CV_strategy,
                                                n_fold=n_fold)
    
            for k, alpha_k in enumerate(alpha):
                quantForest = QuantileRegressionForest(
                                    n_estimators=n_estimators,
                                    min_samples_split=min_samples_leaf_by_dim_and_alpha[k, j]*2,
                                    min_samples_leaf=min_samples_leaf_by_dim_and_alpha[k, j])
                quantForest.fit(X2_j, Y2[boot_idx])
                conditional_quantiles = quantForest.predict(
                                                X=X1_j,
                                                alpha=alpha_k,
                                                method=method,
                                                used_bootstrap_samples=used_bootstrap_samples)
                numerator[j, k] = _averaged_check_function_alpha_float_parallel(
                                                        Y1[boot_idx] - conditional_quantiles.ravel(),
                                                        alpha_k)
        
        qosa_indices[:,:,i] = 1 - numerator/denominator

    return qosa_indices, min_samples_leaf_by_dim_and_alpha




# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------
#
# Computation of the QOSA indices
#
# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------

dim = 2
n_sample = 2*10**4
n_fold = 4
n_boot = 2000

X1 = np.random.exponential(size = (n_sample, 2))
X1[:,1] = -X1[:,1]
Y1 = X1.sum(axis=1)

X2 = np.random.exponential(size = (n_sample, 2))
X2[:,1] = -X2[:,1]
Y2 = X2.sum(axis=1)

alpha = np.array([0.1, 0.5, 0.7, 0.99])
n_estimators = 10**2
min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(800), 30, endpoint=True, dtype=int))

from qosa.tests import DifferenceExponential
model = DifferenceExponential()
model.alpha = alpha
True_values = model.qosa_indices

import time
start = time.time()
qosa, optim_min_samples_leaf =_qosa_forest_compute_quantile(
                                    X1=X1,
                                    X2=X2, 
                                    Y1=Y1,
                                    Y2=Y2,
                                    dim=2,
                                    alpha=alpha,
                                    method="Weighted_CDF",
                                    n_estimators=n_estimators,
                                    min_samples_leaf=min_samples_leaf,
                                    used_bootstrap_samples=False,
                                    optim_by_CV=True,
                                    CV_strategy="K_Fold",
                                    n_fold=n_fold,
                                    n_boot=n_boot)
print(time.time()-start)

# ----------------------------------------------------------------------------
# Percentile CI
# ----------------------------------------------------------------------------

ci_prob = [2.5, 97.5]
CI = np.percentile(qosa[:,:,1:], ci_prob, axis=2)

# ----------------------------------------------------------------------------
# Bias corrected CI
# ----------------------------------------------------------------------------

qosa_estimate = qosa[:,:,0]
qosa_bootstrap = qosa[:,:,1:]

ci_prob = 0.05
z_alpha = stats.norm.ppf(ci_prob*0.5)

# Quantile of Gaussian of the empirical CDF at the no_boot estimation
z_0 = stats.norm.ppf((qosa_bootstrap <= qosa_estimate.reshape(2, alpha.shape[0],1)).mean(axis=2))

# Quantile func of the empirical bootstrap distribution
tmp_down = stats.norm.cdf(2*z_0 + z_alpha)
tmp_up = stats.norm.cdf(2*z_0 - z_alpha)

ci_down = np.zeros((dim, alpha.shape[0]))
ci_up = np.zeros((dim, alpha.shape[0]))
for i in range(dim):
    for j in range(alpha.shape[0]):
        ci_down[i, j] = np.percentile(qosa_bootstrap[i, j, :], tmp_down[i, j]*100.)
        ci_up[i, j] = np.percentile(qosa_bootstrap[i, j, :], tmp_up[i, j]*100.)

# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------

sns.set_context("talk")

pdf_pages = PdfPages('test_model_difference_expo_bootstrap_distribution.pdf')

for l in range(2):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    k = 0
    for i in range(2):
        for j in range(2):
            ax = axes[i,j]
            sns.kdeplot(qosa[l, k, 1:], shade=True, legend=False, ax=ax, linewidth=3,
                        label=r'$\alpha$ = %0.2f' % (alpha[k], ),color = "#003fff")
            ax.axvline(True_values[k, l], label='True value', color='orange')
            
            if i ==1:
                ax.set_xlabel(r'$S_{%d}^{\alpha}$' % (l+1,))
            if j == 0:
                ax.set_ylabel('QOSA estimate')
            ax.legend()
            ax.set_title('CI = [%0.2f, %0.2f]     BC_CI = [%0.2f, %0.2f]' %
                         (CI[0,l,k], CI[1,l,k], ci_down[l,k], ci_up[l,k]), fontsize=16)
            k +=1
    fig.suptitle(r'Distribution of $S_{%d}^{\alpha}$ with n_sample = %d, n_tree = %d' 
                 ',n_leaf = %d, n_fold = %d, n_boot = %d' % (l+1, n_sample, n_estimators, 
                                               min_samples_leaf.shape[0], n_fold, n_boot),
                 y = 1.02, fontsize=16)
    fig.tight_layout(pad = 1.0)
    
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

pdf_pages.close()