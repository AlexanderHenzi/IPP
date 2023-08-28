"""
Refer to https://github.com/xwshen51/DRIG/blob/main/estimate.py
"""

import numpy as np
from scipy.linalg import sqrtm, norm
import pandas as pd


def est_drig(data, gamma, y_idx=-1, del_idx=None, unif_weight=False):
    """DRIG estimator.

    Args:
        data (list of numpy arrays): a list of data from all environments, where the first element is the observational environment.
        gamma (float): hyperparameter in DRIG.
        y_idx (int, optional): index of the response variable. Defaults to -1.
        del_idx (int, optional): index of the variable to exclude. Defaults to None.
        unif_weight (bool, optional): whether to use uniform weights. Defaults to False.

    Returns:
        numpy array: estimated coefficients.
    """
    if del_idx is None:
        del_idx = y_idx
    ## number of environment
    m = len(data)
    if unif_weight:
        w = [1/m]*m
    else:
        w = [data[e].shape[0] for e in range(m)]
        w = [a/sum(w) for a in w]
    ## gram matrices
    gram_x = [] ## E[XX^T]
    gram_xy = [] ## E[XY]
    for e in range(m):
        data_e = data[e]
        n = data_e.shape[0]
        y = data_e[:, y_idx]
        x = np.delete(data_e, (y_idx, del_idx), 1)
        # x = data_e[:, :-1]
        # y = data_e[:, -1]
        gram_x.append(x.T.dot(x)/n)
        gram_xy.append(x.T.dot(y)/n)
    G = (1 - gamma)*gram_x[0] + gamma*sum([a*b for a,b in zip(gram_x, w)])
    Z = (1 - gamma)*gram_xy[0] + gamma*sum([a*b for a,b in zip(gram_xy, w)])
    return np.linalg.inv(G).dot(Z)

def pop_drig(grams, gamma):
    """DRIG estimator at the population level.

    Args:
        grams (list of numpy arrays): a list of gram matrices.
        gamma (float): hyperparameter.

    Returns:
        numpy array: estimated coefficients.
    """
    m = len(grams)
    gram_x = []
    gram_xy = []
    for e in range(m):
        gram = grams[e]
        gram_x.append(gram[:-1, :-1])
        gram_xy.append(gram[:-1, -1])
    G = (1 - gamma)*gram_x[0] + gamma*sum(gram_x)/m
    Z = (1 - gamma)*gram_xy[0] + gamma*sum(gram_xy)/m
    return np.linalg.inv(G).dot(Z)

def est_drig_adap(data, data_test, y_idx=-1, del_idx=None, unif_weight=False):
    """DRIG-A estimator.

    Args:
        data (list of numpy arrays): a list of data from all environments.
        data_test (numpy array): test data.
    """
    if del_idx is None:
        del_idx = y_idx
    ## training stats
    m = len(data)
    if unif_weight:
        w = [1/m]*m
    else:
        w = [data[e].shape[0] for e in range(m)]
        w = [a/sum(w) for a in w]
    gram_x = [] ## E[XX^T]
    gram_xy = [] ## E[XY]
    for e in range(m):
        data_e = data[e]
        n = data_e.shape[0]
        y = data_e[:, y_idx]
        x = np.delete(data_e, (y_idx, del_idx), 1)
        gram_x.append(x.T.dot(x)/n)
        gram_xy.append(x.T.dot(y)/n)
    # delta_x = sum(gram_x - gram_x[0])/m
    delta_x = sum([a*b for a,b in zip(gram_x - gram_x[0], w)])
    # delta_xy = sum(gram_xy - gram_xy[0])/m
    delta_xy = sum([a*b for a,b in zip(gram_xy - gram_xy[0], w)])
    delta_x_sqrt = sqrtm(delta_x)
    delta_x_sqrt_inv = np.linalg.inv(delta_x_sqrt)
    ## test stats
    n = data_test.shape[0]
    y_te = data_test[:, y_idx]
    x_te = np.delete(data_test, (y_idx, del_idx), 1)
    gram_x_te = x_te.T.dot(x_te)/n
    gram_xy_te = x_te.T.dot(y_te)/n
    sigma_x_te_sqrt = sqrtm(np.linalg.inv(gram_x_te))
    ## calculate adaptive gamma
    mat_mid = sqrtm(delta_x_sqrt @ (gram_x_te - gram_x[0]) @ delta_x_sqrt)
    if not np.isrealobj(mat_mid):
        # print("complex number appears")
        mat_mid = np.real(mat_mid)
    gamma_x = delta_x_sqrt_inv @ mat_mid @ delta_x_sqrt_inv
    gamma_y = (sigma_x_te_sqrt @ gamma_x @ delta_xy).T @ sigma_x_te_sqrt @ (gram_xy_te - gram_xy[0]) / norm(sigma_x_te_sqrt @ gamma_x @ delta_xy, 2)**2
    ## estimator 
    G = gram_x[0] + gamma_x @ delta_x @ gamma_x
    Z = gram_xy[0] + gamma_y * gamma_x @ delta_xy
    return np.linalg.inv(G).dot(Z)

def est_anchor(data, gamma, y_idx=-1, del_idx=None, unif_weight=False):
    """Anchor regression estimator.
    """
    if del_idx is None:
        del_idx = y_idx
    m = len(data)
    if unif_weight:
        w = [1/m]*m
    else:
        w = [data[e].shape[0] for e in range(m)]
        w = [a/sum(w) for a in w]
    gram_x = [] ## E[x^T]
    mu_x = [] ## E[X]E[X^T]
    gram_xy = [] ## E[XY]
    mu_xy = [] ## E[X]E[Y]
    for e in range(m):
        data_e = data[e]
        n = data_e.shape[0]
        y = data_e[:, y_idx]
        x = np.delete(data_e, (y_idx, del_idx), 1)
        x_mean = x.mean(0)
        y_mean = y.mean()
        gram_x.append(x.T.dot(x)/n)
        mu_x.append(np.outer(x_mean, x_mean))
        gram_xy.append(x.T.dot(y)/n)
        mu_xy.append(x_mean*y_mean)
    G = sum([a*b for a,b in zip(gram_x, w)]) + (gamma - 1)*sum([a*b for a,b in zip(mu_x, w)])
    Z = sum([a*b for a,b in zip(gram_xy, w)]) + (gamma - 1)*sum([a*b for a,b in zip(mu_xy, w)])
    return np.linalg.inv(G).dot(Z)

def pop_anchor(grams, mus, gamma):
    """Anchor regression at population.
    """
    m = len(grams)
    gram_x = [] ## E[x^T]
    mu_x = [] ## E[X]E[X^T]
    gram_xy = [] ## E[XY]
    mu_xy = []
    for e in range(m):
        gram = grams[e]
        mu = mus[e]
        gram_x.append(gram[:-1, :-1])
        gram_xy.append(gram[:-1, -1])
        mu_x.append(np.outer(mu[:-1], mu[:-1]))
        mu_xy.append(mu[:-1]*mu[-1])
    G = sum(gram_x) + (gamma - 1)*sum(mu_x)
    Z = sum(gram_xy) + (gamma - 1)*sum(mu_xy)
    return np.linalg.inv(G).dot(Z)
        

def est(data, method="drig", gamma=None, y_idx=-1, del_idx=None, unif_weight=False):
    """General estimation function. 

    Args:
        data (list of numpy arrays): a list of data from all environments, where the first element is the observational environment.
        method (str, optional): estimation method. Defaults to "drig".
        gamma (float): hyperparameter in DRIG.
        y_idx (int, optional): index of the response variable. Defaults to -1.
        del_idx (int, optional): index of the variable to exclude. Defaults to None.
        unif_weight (bool, optional): whether to use uniform weights. Defaults to False.

    Returns:
        numpy array: estimated coefficients.
    """
    if del_idx is None:
        del_idx = y_idx
    if method == "ols_pool":
        ## pooled OLS
        b = est_drig(data, 1, y_idx, del_idx, unif_weight)
    elif method == "ols_obs":
        ## observational OLS
        data = data[0]
        y = data[:, y_idx]
        x = np.delete(data, (y_idx, del_idx), 1)
        b = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
    elif method == "drig":
        b = est_drig(data, gamma, y_idx, del_idx, unif_weight)
    elif method == "anchor":
        b = est_anchor(data, gamma, y_idx, del_idx, unif_weight)
    return b

def test_mse(data, b, y_idx=-1, del_idx=None):
    """Test MSE on a single dataset.
    
    Arguments:
        data (numpy array): test data
    """
    if del_idx is None:
        del_idx = y_idx
    x = np.delete(data, (y_idx, del_idx), 1)
    y = data[:, y_idx]
    y_pred = x.dot(b)
    return ((y - y_pred)**2).mean()

def test_mse_list(data, b, pooled=False, stats_only=False, y_idx=-1, del_idx=None):
    """Test on multiple datasets.
    
    Arguments:
        data (list of numpy arrays): a list of test data.
        pooled (bool, optional): whether to compute the MSE on pooled data. Default to False.
    """
    if del_idx is None:
        del_idx = y_idx
    errors = []
    for i in range(len(data)):
        errors.append(test_mse(data[i], b, y_idx, del_idx))
    if pooled:
        errors.append(test_mse(np.concatenate(data), b, y_idx, del_idx))
    if stats_only:
        return np.mean(errors), np.std(errors), np.max(errors)
    else:
        return errors

def test_mse_pop(gram, b):
    """Population test MSE on a single test.
    
    Arguments:
        gram (numpy array): gram matrix of test data
    """
    return gram[-1, -1] + b.T @ gram[:-1, :-1] @ b - 2 * b.T @ gram[:-1, -1]

def test_mse_list_pop(grams, b):
    errors = []
    for i in range(len(grams)):
        errors.append(test_mse_pop(grams[i], b))
    return errors

def eval_test(b, method, results, perturb_stren, test_grams, train_id):
    num_test_envs = len(test_grams)
    return pd.concat([results, pd.DataFrame({
        "train_id": np.repeat(train_id, num_test_envs),
        "method": np.repeat(method, num_test_envs),
        "perturb_stren": np.repeat(perturb_stren, num_test_envs),
        "test_mse": np.array(test_mse_list_pop(test_grams, b)),
    })], ignore_index=True)
    
def est_oracle_gamma(data_train, method, gram_test, gamma_l=0, gamma_u=1000, gamma_step=1):
    """Find the best gamma based on test performance
    
    Arguments:
        data_train (list of numpy arrays): list of finite sample training data
        method (str): "drig" or "anchor"
        gram_test (numpy array): gram matrix of test data
    """
    
    gamma = gamma_l
    while gamma < gamma_u:
        gamma_p = gamma
        error_p = test_mse_pop(gram_test, est(data_train, method, gamma_p))
        gamma += gamma_step
        error = test_mse_pop(gram_test, est(data_train, method, gamma))
        if error > error_p:
            return gamma_p, error_p
    return gamma, error

def est_oracle_gamma_list(data_train, method, grams_test, gamma_l=0, gamma_u=1000, gamma_step=1):
    """Apply `est_oracle_gamma` to a list of test gram matrices
    """
    gammas = []; errors = []
    for gram_test in grams_test:
        gamma, error = est_oracle_gamma(data_train, method, gram_test, gamma_l, gamma_u, gamma_step)
        gammas.append(gamma); errors.append(error)
    return gammas, errors

def eval_test_oracle_gamma(data_train, method, results, perturb_stren, test_grams, gamma_l=0, gamma_u=1000, gamma_step=1, train_id=0):
    gammas, errors = est_oracle_gamma_list(data_train, method, test_grams, gamma_l, gamma_u, gamma_step)
    num_test_envs = len(test_grams)
    method_name = "DRIG oracle" if method == "drig" else "anchor regression oracle"
    return pd.concat([results, pd.DataFrame({
        "train_id": np.repeat(train_id, num_test_envs),
        "method": np.repeat(method_name, num_test_envs),
        "perturb_stren": np.repeat(perturb_stren, num_test_envs),
        "test_mse": np.array(errors),
        "gamma": np.array(gammas)
    })], ignore_index=True)