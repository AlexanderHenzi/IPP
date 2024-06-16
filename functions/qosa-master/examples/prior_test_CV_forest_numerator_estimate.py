# -*- coding: utf-8 -*-

import numpy as np 
from qosa import MinimumConditionalExpectedCheckFunctionForest

n_sample = 2*10**4

X = np.random.exponential(size = (n_sample, 2))
Y1 = X[:,0] - X[:,1]
X1 = X[:,0]

# Second sample to estimate the index
X2 = np.random.exponential(size = n_sample)

alpha = np.array([0.3, 0.5, 0.7])
n_estimators = 10**2
min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(1500), 40, endpoint=True, dtype=int))
numerator = np.zeros((min_samples_leaf.shape[0], alpha.shape[0]))

for i in range(min_samples_leaf.shape[0]):
    print(i)

    min_expectation = MinimumConditionalExpectedCheckFunctionForest(n_estimators=n_estimators,
                                                                    min_samples_leaf=min_samples_leaf[i],
                                                                    min_samples_split=min_samples_leaf[i]*2)
    min_expectation.fit(X1, Y1)
    minimum_expectation = min_expectation.predict(X=X2,
                                                  alpha=alpha)
    numerator[i, :] = minimum_expectation.mean(axis=0)

print('Weighted minimum forest \n')
print(numerator)