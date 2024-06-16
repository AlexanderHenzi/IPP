# -*- coding: utf-8 -*-

import numpy as np
import openturns as ot

from scipy import stats

# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------
#
# Code needed to compute the Shapley effects
#
# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Generating the training sample 
# -----------------------------------------------------------------------------

def _condMVN_new(cov, dependent_ind, given_ind, X_given):
    
    cov = np.asarray(cov)
    
    B = cov[:, dependent_ind]
    B = B[dependent_ind]
    
    C = cov[:, dependent_ind]
    C = C[given_ind]
    
    D = cov[:, given_ind]
    D = D[given_ind]
    
    CDinv = np.dot(np.transpose(C), np.linalg.inv(D))
    
    condMean = np.dot(CDinv, X_given)
    condVar = B - np.dot(CDinv, C)
    condVar = ot.CovarianceMatrix(condVar)
    
    return condMean, condVar


def _cond_sampling_new(distribution, n_sample, idx, idx_c, x_cond):
    margins_dep = [distribution.getMarginal(int(i)) for i in idx]
    margins_cond = [distribution.getMarginal(int(i)) for i in idx_c]

    # Creates a conditioned variables that follows a Normal distribution
    u_cond = np.zeros(x_cond.shape)
    for i, marginal in enumerate(margins_cond):
        u_cond[i] = np.asarray(ot.Normal().computeQuantile(marginal.computeCDF(x_cond[i])))

    sigma = np.asarray(distribution.getCopula().getCorrelation())
    cond_mean, cond_var = _condMVN_new(sigma, idx, idx_c, u_cond)
    
    n_dep = len(idx)
    dist_cond = ot.Normal(cond_mean, cond_var)
    sample_norm = np.asarray(dist_cond.getSample(int(n_sample)))
    sample_x = np.zeros((n_sample, n_dep))
    phi = lambda x: ot.Normal().computeCDF(x)
    for i in range(n_dep):
        u_i = np.asarray(phi(sample_norm[:, i].reshape(-1, 1))).ravel()
        sample_x[:, i] = np.asarray(margins_dep[i].computeQuantile(u_i)).ravel()

    return sample_x


def _sub_sampling(distribution, n_sample, idx):
    # Margins of the subset
    margins_sub = [distribution.getMarginal(int(j)) for j in idx]
    # Get the correlation matrix
    sigma = np.asarray(distribution.getCopula().getCorrelation())
    # Takes only the subset of the correlation matrix
    copula_sub = ot.NormalCopula(ot.CorrelationMatrix(sigma[:, idx][idx, :]))
    # Creates the subset distribution
    dist_sub = ot.ComposedDistribution(margins_sub, copula_sub)
    # Sample
    sample = np.asarray(dist_sub.getSample(int(n_sample)))
    return sample

        
def build_sample(model, input_distribution, n_var, n_outer, n_inner):
    """
    Creates the input and output sample for the computation.
    
    Parameters
    ----------
    model : callable
        The input model function.
               
    n_var : int
        The sample size for the output variance estimation.
        
    n_outer : int
        The number of conditionnal variance estimations.
        
    n_inner : int
        The sample size for the conditionnal output variance estimation.
    """        
    
    dim = input_distribution.getDimension()
    
    # All permutations used to compute the index
    perms = list(ot.KPermutations(dim, dim).generate())
    n_perms = len(perms)
    
    # Creation of the design matrix
    input_sample_1 = np.asarray(input_distribution.getSample(n_var))
    input_sample_2 = np.zeros((n_perms * (dim - 1) * n_outer * n_inner, dim))

    for i_p, perm in enumerate(perms):
        idx_perm_sorted = np.argsort(perm)  # Sort the variable ids
        for j in range(dim - 1):
            # Normal set
            idx_j = perm[:j + 1]
            # Complementary set
            idx_j_c = perm[j + 1:]
            sample_j_c = _sub_sampling(input_distribution, n_outer, idx_j_c)
            for l, xjc in enumerate(sample_j_c):
                # Sampling of the set conditionally to the complementary
                # element
                xj = _cond_sampling_new(input_distribution, n_inner, idx_j, idx_j_c, xjc)
                xx = np.c_[xj, [xjc] * n_inner]
                ind_inner = i_p * (dim - 1) * n_outer * n_inner + j * n_outer * n_inner + l * n_inner
                input_sample_2[ind_inner:ind_inner + n_inner, :] = xx[:, idx_perm_sorted]

    # Model evaluation
    input_sample = np.r_[input_sample_1, input_sample_2]
    output_sample = model(input_sample)
            
    output_sample_1 = output_sample[:n_var]
    output_sample_2 = output_sample[n_var:].reshape((n_perms, dim-1, n_outer, n_inner))
    
    results = {'perms':perms,
               'output_sample_1':output_sample_1,
               'output_sample_2':output_sample_2}
    
    return results


# -----------------------------------------------------------------------------
# Function computing the Shapley effects
# -----------------------------------------------------------------------------

def compute_indices(dim, perms, output_sample_1, output_sample_2, n_var, n_outer, n_boot=1):
    """
    Computes the Shapley indices.
    """
    
    n_perms = len(perms)

    # Initialize Shapley effects for all players
    shapley_indices = np.zeros((dim, n_boot))
    c_hat = np.zeros((n_perms, dim, n_boot))
    
    variance = np.zeros((n_boot))
    perms = np.asarray(perms)

    for i in range(n_boot):
        # Bootstrap sample indexes
        # The first iteration is computed over the all sample.
        if i > 0:
            boot_var_idx = np.random.randint(0, n_var, size=(n_var, ))
            boot_No_idx = np.random.randint(0, n_outer, size=(n_outer, ))
        else:
            boot_var_idx = range(n_var)
            boot_No_idx = range(n_outer)
            
        # Output variance
        var_y = output_sample_1[boot_var_idx].var(axis=0, ddof=1)
        variance[i] = var_y

        # Conditional variances
        output_sample_2 = output_sample_2[:, :, boot_No_idx]
        c_var = output_sample_2.var(axis=3, ddof=1)

        # Conditional exceptations
        c_mean_var = c_var.mean(axis=2)

        # Cost estimation
        c_hat[:, :, i] = np.concatenate((c_mean_var, np.repeat(var_y, repeats=n_perms).reshape(-1,1)), axis=1)

    # Cost variation
    delta_c = c_hat.copy()
    delta_c[:, 1:] = c_hat[:, 1:] - c_hat[:, :-1]
    
    for i in range(n_boot):
        # Estimate Shapley, main and total Sobol effects
        for i_p, perm in enumerate(perms):
            # Shapley effect
            shapley_indices[perm, i] += delta_c[i_p, :, i]
                
    output_variance = variance[np.newaxis]
    shapley_indices = shapley_indices / n_perms / output_variance

    return shapley_indices


# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------
#
# Application on additive gaussian model
#
# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Definition of the model
# -----------------------------------------------------------------------------

dim = 3
margins = [ot.Normal()]*(dim-1) + [ot.Normal(0, 2.)]
copula = ot.NormalCopula(dim)
corr_1_2 = 0.
corr_1_3 = 0.
corr_2_3 = 0.7
theta = [corr_1_2, corr_1_3, corr_2_3]
copula.setParameter(theta)
input_distribution = ot.ComposedDistribution(margins, copula)
model_func = lambda x : x.sum(axis=1)

# -----------------------------------------------------------------------------
# Building the training sample
# -----------------------------------------------------------------------------

n_var = 10**4
n_outer = 10**4
n_inner = 3
max_budget = n_var + np.math.factorial(dim) * (dim - 1) * n_outer * n_inner 
print('Max budget:', max_budget)

results = build_sample(model=model_func,
                       input_distribution=input_distribution,
                       n_var=n_var,
                       n_outer=n_outer,
                       n_inner=n_inner)

# -----------------------------------------------------------------------------
# compute the Shapley effects
# -----------------------------------------------------------------------------

Sh = compute_indices(dim=dim,
                     n_var=n_var, 
                     n_outer=n_outer, 
                     n_boot=500,
                     **results)

True_value = [0.11363636, 0.35625 , 0.53011364]

# Classical IC
np.percentile(Sh, [2.5, 97.5], axis=1)

# Improved CI
Sh_estimate = Sh[:,0]
Sh_bootsrap = Sh[:,1:]

ci_prob = 0.05
z_alpha = stats.norm.ppf(ci_prob*0.5)

# Quantile of Gaussian of the empirical CDF at the no_boot estimation
z_0 = stats.norm.ppf((Sh_bootsrap <= Sh_estimate.reshape(-1, 1)).mean(axis=1))

# Quantile func of the empirical bootstrap distribution
tmp_down = stats.norm.cdf(2*z_0 + z_alpha)
tmp_up = stats.norm.cdf(2*z_0 - z_alpha)

ci_down = np.zeros((dim))
ci_up = np.zeros((dim))
for i in range(dim):
    ci_down[i] = np.percentile(Sh_bootsrap[i,:], tmp_down[i]*100.)
    ci_up[i] = np.percentile(Sh_bootsrap[i,:], tmp_up[i]*100.)

