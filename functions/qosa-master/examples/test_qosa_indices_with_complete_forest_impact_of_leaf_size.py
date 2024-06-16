# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from qosa import MinimumBasedQosaIndices, qosa_Weighted_Min_with_complete_forest
from qosa.tests import DifferenceExponential
from matplotlib.backends.backend_pdf import PdfPages # to make several figures in one pdf

    
# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------
#
# Compute the QOSA indices with several leaf sizes
#
# -----------------------------------------------------------------------------
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# -----------------------------------------------------------------------------

model = DifferenceExponential()

n_samples = 5*10**3
n_tree = 10**2
n_RMSE = 10**1
# Size of the leaf nodes
min_samples_leaf = np.unique(np.logspace(np.log10(2), np.log10(800), 10, endpoint=True, dtype=int))
n_min_samples_leaf = min_samples_leaf.shape[0]
alpha = np.array([0.1, 0.3, 0.7, 0.99])
n_alphas = alpha.shape[0]
dim = model.dim
random_state = np.random.randint(low=1, high=10**5, size=n_RMSE, dtype=np.uint32)

qosa_indices_estimates = np.empty((n_RMSE, n_min_samples_leaf, n_alphas, dim), dtype=np.float64)
qosa_minimum = MinimumBasedQosaIndices()
start = time.time()
for i in range(n_RMSE):
    print(i)
    qosa_minimum.build_sample(model=model, n_samples=n_samples, method='Weighted_Min_with_complete_forest')
    
    for j, min_samples_leaf_temp in enumerate(min_samples_leaf):
        method = qosa_Weighted_Min_with_complete_forest(alpha=alpha,
                                                        n_estimators=n_tree,
                                                        min_samples_leaf=min_samples_leaf_temp,
                                                        used_bootstrap_samples=False,
                                                        optim_by_CV=False,
                                                        n_fold=3,
                                                        random_state_Forest=random_state[i])
        
        qosa_results = qosa_minimum.compute_indices(method)
        qosa_indices_estimates[i,j,:,:] = qosa_results.qosa_indices_estimates

print('Elapsed time :', time.time()-start)
qosa_true_values= model.qosa_indices

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

pdf_pages = PdfPages('./model_difference_exponential_qosa_with_complete_forest_n_samples_%d_n_trees_%d_n_min_samples_leaf_%d_n_rmse_%d.pdf' % 
                    (n_samples, n_tree, n_min_samples_leaf, n_RMSE))

################
# Weighted CDF #
################

for i in range(dim):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8), sharex=True)
    l = 0
    for j in range(2):
        for k in range(2):
            ax = axes[j,k]
            
            # -------------------------------
            # Boxplot of the estimated values
            # -------------------------------
            boxplot_figure = ax.boxplot(qosa_indices_estimates[:,:,l,i], 
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
            ax2.axhline(y=qosa_true_values[l, i], linestyle='--', color=colors[1], label=r'$\alpha = %.2f $' %(alpha[l],))
            
            # -----------------------------------
            # Customization of the axes and Title
            # -----------------------------------
            legend = ax2.legend(loc='best', frameon=True, fontsize = 14, handlelength=0, handletextpad=0)
            for item in legend.legendHandles:
                item.set_visible(False)
        
            if j == 1:
                ax.set_xlabel(r'Value of the $min\_samples\_leaf$ parameter', fontsize=14)
                ax.xaxis.set_ticklabels(min_samples_leaf, fontsize=14, rotation=60)
            #ax.set_ylabel('Estimation of the second term', fontsize=16)
            ax.yaxis.set_tick_params(labelsize=14)
            
            l+=1
    fig.suptitle('Estimate of the QOSA index, variable %d' % (i+1,), fontsize = 16, y = 1.02)
    fig.tight_layout(pad=1.0)
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

pdf_pages.close()