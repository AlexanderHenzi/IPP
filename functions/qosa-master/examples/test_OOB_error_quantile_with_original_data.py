# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from scipy.stats import expon
from qosa import QuantileRegressionForest
from matplotlib.backends.backend_pdf import PdfPages # to make several figures in one pdf

# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------
#
# True value of the numerator
#
# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------

def numerator_wrt_X1(alpha):
    q_X2 = expon.ppf(1-alpha)
    return np.exp(-q_X2)*(1 + q_X2) - alpha
    
# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------
#
# Compute the numerator with several estimators
#
# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------

n_samples = 10**4
n_tree = 5*10**2
n_RMSE = 2*10**2

# Size of the leaf nodes
min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(800), 10, endpoint=True, dtype=int))
n_min_samples_leaf = min_samples_leaf.shape[0]
alpha = np.array([0.1, 0.3, 0.7, 0.9])
n_alphas = alpha.shape[0]
random_state = np.random.randint(low=1, high=10**5, size=n_RMSE, dtype=np.uint32)

numerator_wrt_X1_true_values = numerator_wrt_X1(alpha)
numerator_wrt_X1_Averaged_Quantile_Boot = np.empty((n_RMSE, n_min_samples_leaf, n_alphas), dtype=np.float64)
numerator_wrt_X1_Averaged_Quantile_Orig = np.empty_like(numerator_wrt_X1_Averaged_Quantile_Boot)

start = time.time()
for i in range(n_RMSE):
    print(i)
    
    X = np.random.exponential(size = (n_samples,2))
    Y = X[:,0] - X[:,1]
    X1 = X[:,0].reshape(-1,1)
        
    for j, min_samples_leaf_temp in enumerate(min_samples_leaf):
        #####################
        # Averaged Quantile #
        #####################
        
        qrf = QuantileRegressionForest(n_estimators=n_tree,
                                       min_samples_leaf=min_samples_leaf_temp,
                                       min_samples_split=min_samples_leaf_temp*2,
                                       random_state=random_state[i],
                                       n_jobs=-1)
        qrf.fit(X1, Y,
                oob_score_quantile=True,
                alpha=alpha,
                used_bootstrap_samples=True)
        numerator_wrt_X1_Averaged_Quantile_Boot[i,j,:] = qrf.oob_score_quantile_
        
        qrf._set_oob_score_quantile(alpha, used_bootstrap_samples=False)
        numerator_wrt_X1_Averaged_Quantile_Orig[i,j,:] = qrf.oob_score_quantile_

print("elapsed time", time.time()-start)

# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------
#
# Plot with several graphs on one page
#
# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------

x_axis = np.arange(n_min_samples_leaf)+1
colors = sns.color_palette('bright')
medianprops = dict(linewidth=2, color='black')

pdf_pages = PdfPages('./OOB_quantile_error_%d_n_trees_%d_n_min_samples_leaf_%d_n_rmse_%d.pdf' % 
                    (n_samples, n_tree, n_min_samples_leaf, n_RMSE))

###############################
# Averaged Quantile Bootstrap #
###############################

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8), sharex=True)
k=0    
for i in range(2):
    for j in range(2):
        y_min = min(numerator_wrt_X1_Averaged_Quantile_Boot[:,:,k].min(), numerator_wrt_X1_Averaged_Quantile_Orig[:,:,k].min())
        y_max = max(numerator_wrt_X1_Averaged_Quantile_Boot[:,:,k].max(), numerator_wrt_X1_Averaged_Quantile_Orig[:,:,k].max())
        
        ax = axes[i,j]
        ax.set_ylim(y_min - 0.01, y_max + 0.01)
        
        # -------------------------------
        # Boxplot of the estimated values
        # -------------------------------
        boxplot_figure = ax.boxplot(numerator_wrt_X1_Averaged_Quantile_Boot[:,:,k], 
                                    medianprops=medianprops, 
                                    patch_artist=True,
                                    positions=x_axis)
        
        # -------------------------
        # Points of the true values
        # -------------------------  
        ax2 = ax.twinx()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks([]) 
        ax2.axhline(y=numerator_wrt_X1_true_values[k], linestyle='--', color=colors[1], label=r'$\alpha = %.2f $' %(alpha[k],))
        
        # -----------------------------------
        # Customization of the axes and Title
        # -----------------------------------
        legend = ax2.legend(loc='upper center', frameon=True, fontsize = 14, handlelength=0, handletextpad=0)
        for item in legend.legendHandles:
            item.set_visible(False)
    
        if i == 1:
            ax.set_xlabel(r'Value of the $min\_samples\_leaf$ parameter', fontsize=14)
            ax.xaxis.set_ticklabels(min_samples_leaf, fontsize=14)
        #ax.set_ylabel('Estimation of the second term', fontsize=16)
        ax.yaxis.set_tick_params(labelsize=14)
        
        k+=1
fig.suptitle('With the bootstrap samples', fontsize = 16, y = 1.02)
fig.tight_layout(pad=1.0)
pdf_pages.savefig(fig, bbox_inches='tight')
plt.close(fig)

##############################
# Averaged Quantile Original #
##############################

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8), sharex=True)
k=0    
for i in range(2):
    for j in range(2):
        y_min = min(numerator_wrt_X1_Averaged_Quantile_Boot[:,:,k].min(), numerator_wrt_X1_Averaged_Quantile_Orig[:,:,k].min())
        y_max = max(numerator_wrt_X1_Averaged_Quantile_Boot[:,:,k].max(), numerator_wrt_X1_Averaged_Quantile_Orig[:,:,k].max())
        
        ax = axes[i,j]
        ax.set_ylim(y_min - 0.01, y_max + 0.01)
        
        # -------------------------------
        # Boxplot of the estimated values
        # -------------------------------
        boxplot_figure = ax.boxplot(numerator_wrt_X1_Averaged_Quantile_Orig[:,:,k], 
                                    medianprops=medianprops, 
                                    patch_artist=True,
                                    positions=x_axis)
        
        # -------------------------
        # Points of the true values
        # -------------------------  
        ax2 = ax.twinx()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks([]) 
        ax2.axhline(y=numerator_wrt_X1_true_values[k], linestyle='--', color=colors[1], label=r'$\alpha = %.2f $' %(alpha[k],))
        
        # -----------------------------------
        # Customization of the axes and Title
        # -----------------------------------
        legend = ax2.legend(loc='upper center', frameon=True, fontsize = 14, handlelength=0, handletextpad=0)
        for item in legend.legendHandles:
            item.set_visible(False)
            
        if i == 1:
            ax.set_xlabel(r'Value of the $min\_samples\_leaf$ parameter', fontsize=14)
            ax.xaxis.set_ticklabels(min_samples_leaf, fontsize=14)
        #ax.set_ylabel('Estimation of the second term', fontsize=16)
        ax.yaxis.set_tick_params(labelsize=14)
        
        k+=1
fig.suptitle('With the original sample', fontsize = 16, y = 1.02)
fig.tight_layout(pad=1.0)
pdf_pages.savefig(fig, bbox_inches='tight')
plt.close(fig)

pdf_pages.close()