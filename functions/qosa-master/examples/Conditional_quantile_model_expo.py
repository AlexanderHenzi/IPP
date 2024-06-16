# -*- coding: utf-8 -*-

import time
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages # to make several figures in one pdf
#plt.switch_backend('Agg') #################### très important to plot on cluster
#from scipy.stats import expon
#from pyquantregForest import QuantileRegressionForest

################################################################################
# Test for computing the conditional quantile wrt one variable by considering
# the following model Y = X1 - X2 with X1 and X2 two exponential distributions 
################################################################################

# ---------------------------------------
# Training sample to construct the forest
# ---------------------------------------

from sklearn.model_selection import cross_val_score, KFold
from qosa import QuantileRegressionForest, cross_validation

n_sample = 10**4
X = np.random.exponential(size = (n_sample,2))
y = X[:,0] - X[:,1]
X1 = X[:,0].reshape(-1,1)

alpha = np.array([0.4, 0.9])
min_sample_leaf = np.unique(np.logspace(np.log10(5), np.log10(1500), 10, endpoint=True, dtype=int))

# Get the leaf which gives the minimum value for the loss function 
start = time.time()
cross_validation_values = cross_validation(X1, y, alpha, min_sample_leaf, n_splits=4, method='Forest')
cross_validation_values_bis = cross_validation(X1, y, alpha, min_sample_leaf, n_splits=3, objective='averaged_check_function', method='Forest')
print("alpha = {}".format(alpha))
print("time = {}".format(time.time() - start))
print("cross_validation = {}".format(cross_validation_values.mean(axis=1)))

min_sample_leaf = min_sample_leaf[np.argmin(cross_validation_values.mean(axis=1))]
print("min_sample_leaf = {} \n \n".format(min_sample_leaf))


alpha = 0.7
min_sample_leaf = np.unique(np.logspace(np.log10(5), np.log10(3000), 41, endpoint=True, dtype=int))

# Get the leaf which gives the minimum value for the loss function 
start = time.time()
cross_validation_values = cross_validation(X1, y, alpha, min_sample_leaf, n_splits=5)
print("alpha = {}".format(alpha))
print("time = {}".format(time.time() - start))
print("cross_validation = {}".format(cross_validation_values.mean(axis=1)))

min_sample_leaf = min_sample_leaf[np.argmin(cross_validation_values.mean(axis=1))]
print("min_sample_leaf = {} \n \n".format(min_sample_leaf))

alpha = 0.99
min_sample_leaf = np.unique(np.logspace(np.log10(5), np.log10(3000), 41, endpoint=True, dtype=int))

# Get the leaf which gives the minimum value for the loss function 
start = time.time()
cross_validation_values = cross_validation(X1, y, alpha, min_sample_leaf, n_splits=5)
print("alpha = {}".format(alpha))
print("time = {}".format(time.time() - start))
print("cross_validation = {}".format(cross_validation_values.mean(axis=1)))

min_sample_leaf = min_sample_leaf[np.argmin(cross_validation_values.mean(axis=1))]
print("min_sample_leaf = {}".format(min_sample_leaf))


## ------------------------------------------------------------------
## Code for compute the QOSA indices 
## ------------------------------------------------------------------
#n_sample = 10**5
#alpha = 0.7
#min_sample_leaf = 103
#
#def _check_function(u, alpha):
#    """
#    Definition of the check function also called ppinball loss function.
#    """
#    return u*(alpha - (u < 0.))
#
#
#n_min_split = min_sample_leaf*2
#
## Use one sample to calibrate the forest, the quantiles and the indices
#X = np.random.exponential(size = (n_sample,2))
#Y = X[:,0] - X[:,1]
#X1 = X[:,0].reshape(-1,1)
#
#theta = np.percentile(Y, q=alpha*100)
#denominateur = _check_function(Y-theta, alpha).mean()
#
## Use a second sample to calibrate the forest and compute the quantiles
#X_bis = np.random.exponential(size = (n_sample,2))
#Y_bis = X_bis[:,0] - X_bis[:,1]
#X1_bis = X_bis[:,0].reshape(-1,1)
#
#quantForest = QuantileRegressionForest(min_samples_split=n_min_split, min_samples_leaf=min_sample_leaf)
#quantForest.fit(X1_bis, Y_bis)
#conditional_quantiles = quantForest.predict(X1, alpha)
#numerator = _check_function(Y-conditional_quantiles,alpha).mean()
#print(1 - numerator/denominateur)
































# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

# Compute the RMSE of the conditional expectation function of the leaf nodes' size

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

#start = time.time()
#
#np.random.seed(8739471)
#
#n_sample = 10**5
#n_tree = 10**3
#n_RMSE = 10**3
#
## Size of the leaf nodes
#n_min_leaf = np.unique(np.logspace(np.log10(5), np.log10(3000), 41, endpoint=True, dtype=int))
#n_min_split = n_min_leaf*2
#
## Conditional quantiles
#X1_values_quantiles = np.array([0.5, 0.9, 0.99, 0.995])
#X1_values = np.array([0.69, 2.30, 4.60, 5.29])
#
#alpha_values = np.array([0.01, 0.5, 0.75, 0.9])
#qY_X1_values = np.zeros((X1_values.shape[0], alpha_values.shape[0]))
#for i in range(X1_values.shape[0]):
#    qY_X1_values[i,:] = X1_values[i] - expon.ppf(1-alpha_values)  
#estimated_conditional_quantiles = np.zeros((X1_values.shape[0], alpha_values.shape[0], n_min_leaf.shape[0], n_RMSE))
#
#for k in range(n_RMSE):
#    print(k)
#    
#    X = np.random.exponential(size = (n_sample,2))
#    y = X[:,0] - X[:,1]
#    X1 = X[:,0].reshape(-1,1)
#    
#    for j,n_min_leaf_temp in enumerate(n_min_leaf):
#        n_min_split_temp = n_min_split[j]
#        quantForest = QuantileRegressionForest(n_estimators=n_tree, min_samples_split=n_min_split_temp, min_samples_leaf=n_min_leaf_temp, n_jobs=-1)
#        quantForest.fit(X1, y)
#        estimated_conditional_quantiles[:,:,j,k] = quantForest.compute_conditional_quantile(X1_values, alpha_values)
#
#print('Total time is:{}'.format(time.time()-start))
#
## Quantiles
#cond_quantiles = np.zeros((X1_values.shape[0], alpha_values.shape[0], n_min_leaf.shape[0], n_RMSE))
#cond_quantiles_standardized = np.zeros((X1_values.shape[0], alpha_values.shape[0], n_min_leaf.shape[0], n_RMSE))
#
#for i in range(X1_values.shape[0]):
#    for j in range(alpha_values.shape[0]):
#        cond_quantiles[i,j,:,:] = estimated_conditional_quantiles[i,j,:,:] - qY_X1_values[i,j]
#        cond_quantiles_standardized[i,j,:,:] = (estimated_conditional_quantiles[i,j,:,:] - qY_X1_values[i,j])/qY_X1_values[i,j]
#        
#cond_quantiles = cond_quantiles**2
#RMSE_quantiles = np.sqrt(cond_quantiles.mean(axis=3))
#
#cond_quantiles_standardized = cond_quantiles_standardized**2
#RMSE_quantiles_standardized = np.sqrt(cond_quantiles_standardized.mean(axis=3))
#
#
## ---------
## Plot RMSE
## ---------
#
#pdf_pages = PdfPages('./model_expo_X1_RMSE_conditional_quantiles_function_size_min_samples_leaf_with_n_RMSE_%d_number_min_leaf_%d_n_sample_%d_n_tree_%d.pdf' 
#                     % (n_RMSE, n_min_leaf.shape[0], n_sample, n_tree))
#
#
#for i, X1 in enumerate(X1_values):
#
##------------
## RMSE Normal
##------------
#
#    fig, axes = plt.subplots(figsize = (14,8))
#    axes.plot(n_min_leaf, RMSE_quantiles[i,:,:].T, linewidth = 2)
#    axes.set_xlabel('Size min_samples_leaf', fontsize = 16)
#    axes.xaxis.set_tick_params(labelsize = 16)
#    axes.yaxis.set_tick_params(labelsize = 16)
#    axes.legend([r"$\alpha=%.2f$" % (alpha_values.squeeze()[0],),
#                 r"$\alpha=%.1f$" % (alpha_values.squeeze()[1],),
#                 r"$\alpha=%.2f$" % (alpha_values.squeeze()[2],),
#                 r"$\alpha=%.2f$" % (alpha_values.squeeze()[3],)], fontsize = 16)
#    axes.set_title("RMSE of "+ r"$q_{\alpha}[Y|X_{1} = %0.2f] (%0.3f)$" % (X1_values[i],X1_values_quantiles[i])+" with "+
#                   "$N_{sample}=%d, N_{tree}=%d, N_{RMSE}=%d$" % (n_sample, n_tree, n_RMSE), fontsize = 16)
#    fig.tight_layout()
#    pdf_pages.savefig(fig, bbox_inches='tight')
#    plt.close(fig) 
#    
#    fig, axes = plt.subplots(figsize = (14,8))
#    axes.plot(n_min_leaf, RMSE_quantiles[i,:,:].T, linewidth = 2)
#    axes.set_ylim(0., 0.3)
#    axes.set_xlabel('Size min_samples_leaf', fontsize = 16)
#    axes.xaxis.set_tick_params(labelsize = 16)
#    axes.yaxis.set_tick_params(labelsize = 16)
#    axes.legend([r"$\alpha=%.2f$" % (alpha_values.squeeze()[0],),
#                 r"$\alpha=%.1f$" % (alpha_values.squeeze()[1],),
#                 r"$\alpha=%.2f$" % (alpha_values.squeeze()[2],),
#                 r"$\alpha=%.2f$" % (alpha_values.squeeze()[3],)], fontsize = 16)
#    axes.set_title("RMSE of "+ r"$q_{\alpha}[Y|X_{1} = %0.2f] (%0.3f)$" % (X1_values[i],X1_values_quantiles[i])+" with "+
#                   "$N_{sample}=%d, N_{tree}=%d, N_{RMSE}=%d$" % (n_sample, n_tree, n_RMSE), fontsize = 16)
#    fig.tight_layout()
#    pdf_pages.savefig(fig, bbox_inches='tight')
#    plt.close(fig) 
#    
#    #------------
#    # RMSE Standardized
#    #------------
#    
#    fig, axes = plt.subplots(figsize = (14,8))
#    axes.plot(n_min_leaf, RMSE_quantiles_standardized[i,:,:].T, linewidth = 2)
#    axes.set_xlabel('Size min_samples_leaf', fontsize = 16)
#    axes.xaxis.set_tick_params(labelsize = 16)
#    axes.yaxis.set_tick_params(labelsize = 16)
#    axes.legend([r"$\alpha=%.2f$" % (alpha_values.squeeze()[0],),
#                 r"$\alpha=%.1f$" % (alpha_values.squeeze()[1],),
#                 r"$\alpha=%.2f$" % (alpha_values.squeeze()[2],),
#                 r"$\alpha=%.2f$" % (alpha_values.squeeze()[3],)], fontsize = 16)
#    axes.set_title("RMSE (Standardized) of "+ r"$q_{\alpha}[Y|X_{1} = %0.2f] (%0.3f)$" % (X1_values[i],X1_values_quantiles[i])+" with "+
#                   "$N_{sample}=%d, N_{tree}=%d, N_{RMSE}=%d$" % (n_sample, n_tree, n_RMSE), fontsize = 16)
#    fig.tight_layout()
#    pdf_pages.savefig(fig, bbox_inches='tight')
#    plt.close(fig) 
#    
#    fig, axes = plt.subplots(figsize = (14,8))
#    axes.plot(n_min_leaf, RMSE_quantiles_standardized[i,:,:].T, linewidth = 2)
#    axes.set_ylim(0., 0.3)
#    axes.set_xlabel('Size min_samples_leaf', fontsize = 16)
#    axes.xaxis.set_tick_params(labelsize = 16)
#    axes.yaxis.set_tick_params(labelsize = 16)
#    axes.legend([r"$\alpha=%.2f$" % (alpha_values.squeeze()[0],),
#                 r"$\alpha=%.1f$" % (alpha_values.squeeze()[1],),
#                 r"$\alpha=%.2f$" % (alpha_values.squeeze()[2],),
#                 r"$\alpha=%.2f$" % (alpha_values.squeeze()[3],)], fontsize = 16)
#    axes.set_title("RMSE (Standardized) of "+ r"$q_{\alpha}[Y|X_{1} = %0.2f] (%0.3f)$" % (X1_values[i],X1_values_quantiles[i])+" with "+
#                   "$N_{sample}=%d, N_{tree}=%d, N_{RMSE}=%d$" % (n_sample, n_tree, n_RMSE), fontsize = 16)
#    fig.tight_layout()
#    pdf_pages.savefig(fig, bbox_inches='tight')
#    plt.close(fig) 
#
## Boxplot
#
#for i,X1 in enumerate(X1_values):
#    for j, alpha in enumerate(alpha_values):
#        
#        fig, axes = plt.subplots(figsize=(14,8))
#        axes.boxplot(estimated_conditional_quantiles[i,j,:,:].T, showmeans=True)
#        axes.axhline(qY_X1_values[i,j], linestyle='--')
#        axes.set_xlabel('Values of min_samples_leaf', fontsize=14)
#        axes.xaxis.set_ticklabels(n_min_leaf, fontsize=14, rotation = 60)
#        axes.yaxis.set_tick_params(labelsize = 14)
#        axes.set_title("$q_{%0.2f} [Y | X_1 = %0.2f]$" %(alpha_values[j], X1_values.squeeze()[i]), fontsize = 14)
#        fig.tight_layout()
#        pdf_pages.savefig(fig, bbox_inches='tight')
#        plt.close(fig)    
#
#pdf_pages.close()