# -*- coding: utf-8 -*-

import warnings
import numbers
from traceback import format_exception_only

import numpy as np

from sklearn.utils import indexable
from sklearn.utils.metaestimators import _safe_split
from sklearn.metrics.scorer import check_scoring
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection._split import check_cv

import numba as nb
from time import perf_counter
from sklearn.model_selection import KFold

from qosa import QuantileRegressionForest
from qosa import cross_validate



## ----------------------------------------------------------------------------
## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## ----------------------------------------------------------------------------
##
## _VALIDATION FILE
##
## ----------------------------------------------------------------------------
## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## ----------------------------------------------------------------------------

def cross_validate_test(estimator, X, y, cv, scoring, scoring_params=None, error_score=np.nan):
    """
    Evaluate metric(s) by cross-validation.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like
        The data to fit.

    y : array-like
        The target variable to try to predict.

    cv : cross-validation generator
        Determines the cross-validation splitting strategy.
        
    scoring : callable
        callable to evaluate the predictions on the test set.
        
    scoring_params : dict, optional
        Additional parameters to pass to the scoring function.
       
    error_score : 'numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.
        
    Returns
    -------
    res_scorer : arrays of shape=(n_splits,)
        Array of scores of the estimator for each run of the cross validation.
    """

    # Part to verify the data    
    X, y = indexable(X, y)
    cv = check_cv(cv=cv, y=y)
    scorer = check_scoring(estimator, scoring=scoring)
    
    res_scorer = list()
    # Compute the scoring function for each fold
    for train, test in cv.split(X, y):
        res_scorer.append(_fit_and_score(estimator, X, y, scorer, scoring_params, train, test, error_score=error_score))
    
    res_scorer = np.array(res_scorer)
    
    return res_scorer


def _fit_and_score(estimator, X, y, scorer, scoring_params, train, test, error_score=np.nan):
    """
    Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like, optional, default: None
        The target variable to try to predict.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        The callable object should have signature
        ``scorer(estimator, X, y)``.
    
    scoring_params : dict or None
        Additional parameters to pass to the scorer function.

    train : array-like, shape (n_train_samples,)
        Indices of training samples.

    test : array-like, shape (n_test_samples,)
        Indices of test samples.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.

    Returns
    -------

    """
    
    scoring_params = scoring_params if scoring_params is not None else {}
    
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    try:
        estimator.fit(X_train, y_train)
    except Exception as e:
        test_scores = [error_score]
        warnings.warn("Estimator fit failed. The score on this train-test"
                      " partition for these parameters will be set to %f. "
                      "Details: \n%s" %
                      (error_score, format_exception_only(type(e), e)[0]),
                      FitFailedWarning)
    else:
        test_scores = _score(estimator, X_test, y_test, scorer, scoring_params)
    
    return test_scores


def _score(estimator, X_test, y_test, scorer, scoring_params):
    """
    Compute the score(s) of an estimator on a given test set.
    """

    score = scorer(estimator, X_test, y_test, **scoring_params)

    if isinstance(score, np.ndarray):
        for i in range(len(score)):
            if not isinstance(score[i], numbers.Number):
                raise ValueError("scoring must return a number, got %s (%s) "
                                 "instead. (scorer=%r)"
                                 % (str(score), type(score), scorer))
    return score



# -----------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# -----------------------------------------------------------------------------
#
# CROSS VALIDATION FILE
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


np.random.seed(0)
n_sample = 10**4
X = np.random.exponential(size = (n_sample,2))
y = X[:,0] - X[:,1]
X = X[:,0].reshape(-1,1)

min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(1000), 5, endpoint=True, dtype=int))
n_estimators = 2*10**2
n_splits = 4
shuffle = True

i=4
estimator = QuantileRegressionForest(n_estimators=n_estimators, min_samples_split=min_samples_leaf[i]*2, min_samples_leaf=min_samples_leaf[i], random_state=0)
scoring = check_function_error
cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=0)

scoring_params = {'alpha':np.array([0.1, 0.5, 0.9]), 'used_bootstrap_samples':False}
a = cross_validate_test(estimator, X, y=y, cv=cv, scoring=scoring, scoring_params=scoring_params)
b = cross_validate(estimator, X, y=y, cv=cv, scoring=scoring, scoring_params=scoring_params)