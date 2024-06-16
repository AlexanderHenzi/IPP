# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 15:51:53 2020

@author: U005032
"""

import numba as nb 
import numpy as np
import openturns as ot

@nb.njit("float64(float64[:], float64)", nogil=True, cache=False, parallel=False)
def _averaged_check_function_alpha_float_unparallel(u, alpha):
    """
    Definition of the check function also called pinball loss function.
    """
    
    n_samples = u.shape[0]
    res = 0.    
    for i in range(n_samples):
        res += u[i]*(alpha - (u[i] < 0.))
    
    res /= n_samples
    return res

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

@nb.njit("float64[:](float64[:], float64[:])", nogil=True, cache=False, parallel=False)
def _compute_first_term_Browne_estimate_1(Y, alpha):
    n_alphas = alpha.shape[0]
    argsort_Y = np.argsort(Y)
    expectation = np.empty(n_alphas, dtype=np.float64)
    
    j=0
    for i, alpha_temp in enumerate(alpha):
        if j < Y.shape[0]:
            ans = _averaged_check_function_alpha_float_unparallel(Y - Y[argsort_Y[j]], alpha_temp) 
        else:
            for k in range(i, n_alphas):
                expectation[k] = _averaged_check_function_alpha_float_unparallel(Y - Y[argsort_Y[-1]], alpha[k])
            break
                
        if i == 0:
            j=1
        else:
            j+=1
        while(j < Y.shape[0]):
            temp = _averaged_check_function_alpha_float_unparallel(Y - Y[argsort_Y[j]], alpha_temp)
            if(temp <= ans):
                ans = temp
                j+=1
            else:
                j-=1
                break
        expectation[i] = ans
    
    return expectation

@nb.njit("float64[:](float64[:], float64[:])", nogil=True, cache=False, parallel=False)
def _compute_first_term_Browne_estimate_2(Y, alpha):
    n_alphas = alpha.shape[0]
    argsort_Y = np.argsort(Y)
    expectation = np.empty(n_alphas, dtype=np.float64)
    
    j=0
    for i, alpha_temp in enumerate(alpha):
        if j < Y.shape[0]:
            ans = _averaged_check_function_alpha_float_parallel(Y - Y[argsort_Y[j]], alpha_temp) 
        else:
            for k in range(i, n_alphas):
                expectation[k] = _averaged_check_function_alpha_float_parallel(Y - Y[argsort_Y[-1]], alpha[k])
            break
                
        if i == 0:
            j=1
        else:
            j+=1
        while(j < Y.shape[0]):
            temp = _averaged_check_function_alpha_float_parallel(Y - Y[argsort_Y[j]], alpha_temp)
            if(temp <= ans):
                ans = temp
                j+=1
            else:
                j-=1
                break
        expectation[i] = ans
    
    return expectation

@nb.njit("float64[:](float64[:], float64[:])", nogil=True, cache=False, parallel=True)
def _compute_first_term_Browne_estimate_3(Y, alpha):
    n_alphas = alpha.shape[0]
    argsort_Y = np.argsort(Y)
    expectation = np.empty(n_alphas, dtype=np.float64)
    
    for i in nb.prange(n_alphas):
        alpha_temp = alpha[i]
        ans = _averaged_check_function_alpha_float_unparallel(Y - Y[argsort_Y[0]], alpha_temp) 
        m=1
        while(m < Y.shape[0]):
            temp = _averaged_check_function_alpha_float_unparallel(Y - Y[argsort_Y[m]], alpha_temp)
            if(temp <= ans):
                ans = temp
                m+=1
            else:
                break
        expectation[i] = ans
    
    return expectation

# n_sample = 10**5
# alpha = np.array([0.1, 0.5, 0.9])

# # First sample to compute the conditional CDF
# X = np.random.exponential(size = (n_sample, 2))
# Y1 = X[:,0] - X[:,1]

# import cProfile
# import pstats

# print('method 1')
# profiler = cProfile.Profile(builtins = False)
# profiler.enable()
# for i in range(10):
#     a = _compute_first_term_Browne_estimate_1(Y1, alpha)
# profiler.disable()
# profiler.dump_stats('normal_profiler')
# pstats.Stats('normal_profiler').strip_dirs().sort_stats('time').print_stats(10)

# print('method 2')
# profiler = cProfile.Profile(builtins = False)
# profiler.enable()
# for i in range(10):
#     b = _compute_first_term_Browne_estimate_2(Y1, alpha)
# profiler.disable()
# profiler.dump_stats('normal_profiler')
# pstats.Stats('normal_profiler').strip_dirs().sort_stats('time').print_stats(10)

# print('method 3')
# profiler = cProfile.Profile(builtins = False)
# profiler.enable()
# for i in range(10):
#     c = _compute_first_term_Browne_estimate_3(Y1, alpha)
# profiler.disable()
# profiler.dump_stats('normal_profiler')
# pstats.Stats('normal_profiler').strip_dirs().sort_stats('time').print_stats(10)

# print((a != b).sum())
# print((b != c).sum())
# print((a != c).sum())

###############################################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
###############################################################################

# Estimation of the second term in the index

@nb.njit("void(float64[:], float64, float64[:], float64)", nogil=True, cache=False, parallel=False)
def _gaussian_kernel(kernel_evaluation, x, input_samples, bandwidth):

    pi = np.sqrt(2*np.pi)
    for i in range(input_samples.shape[0]):
        kernel_evaluation[i] = np.exp(-0.5*((x - input_samples[i])/bandwidth)**2)/pi


@nb.njit("float64(float64[:], float64[:], float64, float64, float64, float64)", nogil=True, cache=False, parallel=False)
def _averaged_check_function_alpha_float_unparallel_product_kernel(output_samples, kernel, X_density_i, bandwidth, theta, alpha):
    
    n_samples = output_samples.shape[0]
    res = 0.    
    for i in range(n_samples):
        res += (output_samples[i] - theta)*(alpha - ((output_samples[i] - theta) < 0.))*kernel[i]
    
    res /= (n_samples*X_density_i*bandwidth)
    return res


@nb.njit("void(float64[:,:], float64[:], float64[:], float64[:], float64[:], float64[:], float64)", nogil=True, cache=False, parallel=True)
def _compute_second_term_Browne_estimate(minimum_expectation, input_samples, output_samples, X, X_density, alpha, bandwidth):
    
    n_samples = input_samples.shape[0]
    n_quantiles = X.shape[0]
    argsort_output_samples = np.argsort(output_samples)
    
    for i in nb.prange(n_quantiles):
        kernel = np.empty(n_samples, dtype=np.float64)
        _gaussian_kernel(kernel, X[i], input_samples, bandwidth)
        
        k=0
        for j, alpha_temp in enumerate(alpha):
            if k < n_samples:
                ans = _averaged_check_function_alpha_float_unparallel_product_kernel(
                                                     output_samples,
                                                     kernel,
                                                     X_density[i],
                                                     bandwidth,
                                                     output_samples[argsort_output_samples[k]],
                                                     alpha_temp)
            else:
                for l in range(j, alpha.shape[0]):
                    minimum_expectation[i, l] = _averaged_check_function_alpha_float_unparallel_product_kernel(
                                                         output_samples,
                                                         kernel,
                                                         X_density[i],
                                                         bandwidth,
                                                         output_samples[argsort_output_samples[-1]],
                                                         alpha[l])
                break
            
            if j == 0:
                k=1
            else:
                k+=1
            while(k < n_samples):
                temp = _averaged_check_function_alpha_float_unparallel_product_kernel(
                                                  output_samples,
                                                  kernel,
                                                  X_density[i],
                                                  bandwidth,
                                                  output_samples[argsort_output_samples[k]],
                                                  alpha_temp)
                if(temp <= ans):
                    ans = temp
                    k+=1
                else:
                    k-=1
                    break
            minimum_expectation[i, j] = ans
#            print("alpha: ", alpha_temp, output_samples[argsort_output_samples[m-1]])


@nb.njit("void(float64[:,:], float64[:], float64[:], float64[:], float64[:], float64[:], float64)", nogil=True, cache=False, parallel=False)
def _compute_second_term_Browne_estimate_second(minimum_expectation, input_samples, output_samples, X, X_density, alpha, bandwidth):
    
    n_samples = input_samples.shape[0]
    n_quantiles = X.shape[0]
    n_alphas = alpha.shape[0]
    argsort_output_samples = np.argsort(output_samples)
    
    for i in nb.prange(n_quantiles):
        
        kernel = np.empty(n_samples, dtype=np.float64)
        _gaussian_kernel(kernel, X[i], input_samples, bandwidth)
        
        k = 0
        alpha_temp = alpha[k]
        ans = _averaged_check_function_alpha_float_unparallel_product_kernel(
                                                 output_samples,
                                                 kernel,
                                                 X_density[i],
                                                 bandwidth,
                                                 output_samples[argsort_output_samples[0]],
                                                 alpha_temp)
        
        for j in range(1, n_samples):
            temp = _averaged_check_function_alpha_float_unparallel_product_kernel(
                                                  output_samples,
                                                  kernel,
                                                  X_density[i],
                                                  bandwidth,
                                                  output_samples[argsort_output_samples[j]],
                                                  alpha_temp)
            if(temp <= ans):
                ans = temp
            else:
                minimum_expectation[i, k] = ans
#                print("alpha: ", alpha_temp, output_samples[argsort_output_samples[j-1]])
                k += 1
                if k != n_alphas:
                    alpha_temp = alpha[k]
                    ans = _averaged_check_function_alpha_float_unparallel_product_kernel(
                                                 output_samples,
                                                 kernel,
                                                 X_density[i],
                                                 bandwidth,
                                                 output_samples[argsort_output_samples[j]],
                                                 alpha_temp)
                else: 
                    break


@nb.njit("void(float64[:,:], float64[:], float64[:], float64[:], float64[:], float64[:], float64)", nogil=True, cache=False, parallel=True)
def _compute_second_term_Browne_estimate_third(minimum_expectation, input_samples, output_samples, X, X_density, alpha, bandwidth):
    
    n_samples = input_samples.shape[0]
    n_quantiles = X.shape[0]
    n_alphas = alpha.shape[0]
    argsort_output_samples = np.argsort(output_samples)
    
    for i in nb.prange(n_quantiles):
        
        kernel = np.empty(n_samples, dtype=np.float64)
        _gaussian_kernel(kernel, X[i], input_samples, bandwidth)
        
        k = 0
        alpha_temp = alpha[k]
        ans = _averaged_check_function_alpha_float_unparallel_product_kernel(
                                                 output_samples,
                                                 kernel,
                                                 X_density[i],
                                                 bandwidth,
                                                 output_samples[argsort_output_samples[0]],
                                                 alpha_temp)
        
        l = 1
        while(k < n_alphas):
            temp = _averaged_check_function_alpha_float_unparallel_product_kernel(
                                                  output_samples,
                                                  kernel,
                                                  X_density[i],
                                                  bandwidth,
                                                  output_samples[argsort_output_samples[l]],
                                                  alpha_temp)
            if(temp <= ans):
                ans = temp
                l += 1
            else:
                minimum_expectation[i, k] = ans
                k += 1
                
                if k != n_alphas:
                    alpha_temp = alpha[k]
                    l -= 1
                    ans = _averaged_check_function_alpha_float_unparallel_product_kernel(
                                                 output_samples,
                                                 kernel,
                                                 X_density[i],
                                                 bandwidth,
                                                 output_samples[argsort_output_samples[l]],
                                                 alpha_temp)


n_sample = 10**3
alpha = np.array([0.3, 0.5, 0.7])

# First sample to compute the conditional CDF
X = np.random.exponential(size = (n_sample, 2))
Y1 = X[:,0] - X[:,1]

X_value = np.random.exponential(size = (n_sample))
density = np.array(ot.Exponential().computePDF(X_value.reshape(-1, 1))).ravel()

import cProfile
import pstats

print('method 1')
profiler = cProfile.Profile(builtins = False)
profiler.enable()
minimum_expectation_1 = np.zeros((X_value.shape[0], alpha.shape[0]), 
                                dtype=np.float64,
                                order='C')
for i in range(10):
    _compute_second_term_Browne_estimate(minimum_expectation_1, X[:,0], Y1, X_value, density, alpha, (n_sample)**(-0.2))
a = minimum_expectation_1.mean(axis=0)
profiler.disable()
profiler.dump_stats('normal_profiler')
pstats.Stats('normal_profiler').strip_dirs().sort_stats('time').print_stats(10)

print('method 2')
profiler = cProfile.Profile(builtins = False)
profiler.enable()
minimum_expectation_2 = np.zeros((X_value.shape[0], alpha.shape[0]), 
                                dtype=np.float64,
                                order='C')
for i in range(10):
    _compute_second_term_Browne_estimate_second(minimum_expectation_2, X[:,0], Y1, X_value, density, alpha, (n_sample)**(-0.2))
b = minimum_expectation_2.mean(axis=0)
profiler.disable()
profiler.dump_stats('normal_profiler')
pstats.Stats('normal_profiler').strip_dirs().sort_stats('time').print_stats(10)

print('method 3')
profiler = cProfile.Profile(builtins = False)
profiler.enable()
minimum_expectation_3 = np.zeros((X_value.shape[0], alpha.shape[0]), 
                                dtype=np.float64,
                                order='C')
for i in range(10):
    _compute_second_term_Browne_estimate_third(minimum_expectation_3, X[:,0], Y1, X_value, density, alpha, (n_sample)**(-0.2))
c = minimum_expectation_3.mean(axis=0)
profiler.disable()
profiler.dump_stats('normal_profiler')
pstats.Stats('normal_profiler').strip_dirs().sort_stats('time').print_stats(10)


(minimum_expectation_1 != minimum_expectation_2).sum(axis=0)

np.where(minimum_expectation_1[:,2] != minimum_expectation_2[:,2])

print((a != b).sum())
print((b != c).sum())
print((a != c).sum())



