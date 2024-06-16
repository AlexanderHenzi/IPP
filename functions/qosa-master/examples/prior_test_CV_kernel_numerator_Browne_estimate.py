# -*- coding: utf-8 -*-

import numpy as np 
import openturns as ot

from qosa import MinimumConditionalExpectedCheckFunctionKernel

n_sample = 2*10**4

X = np.random.exponential(size = (n_sample, 2))
Y1 = X[:,0] - X[:,1]
X1 = X[:,0]

# Second sample to estimate the index
X2 = np.random.exponential(size = n_sample)

alpha = np.array([0.3, 0.5, 0.7])
bandwidth = np.linspace(0.001, 0.10, 60, endpoint=True, dtype=np.float64)
numerator = np.zeros((bandwidth.shape[0], alpha.shape[0]))

density_i = np.array(ot.Exponential().computePDF(X2.reshape(-1, 1))).ravel()
min_expectation = MinimumConditionalExpectedCheckFunctionKernel()
min_expectation.fit(X1, Y1)

for i in range(bandwidth.shape[0]):
    print(i)
    minimum_expectation = min_expectation.predict(X=X2,
                                                  X_density=density_i,
                                                  alpha=alpha,
                                                  bandwidth=bandwidth[i])
    numerator[i, :] = minimum_expectation.mean(axis=0)


print('Browne estimate \n')
print(numerator)