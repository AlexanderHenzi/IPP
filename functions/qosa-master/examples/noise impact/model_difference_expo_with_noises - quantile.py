# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
from scipy.stats import laplace, norm
from tqdm import tqdm


from qosa import QuantileBasedQosaIndices, qosa_Quantile__Averaged_Quantile
from qosa.tests import BiDifferenceExponential


# ------------------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ------------------------------------------------------------------------------------
#
# Compute the QOSA indices with two different methods based on minimum
#
# ------------------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ------------------------------------------------------------------------------------

model = BiDifferenceExponential()
dim = model.dim

seed = 888
rng = np.random.default_rng(seed)
n_RMSE = 10**2
n_samples = 10**4
n_trees = 10**2
min_samples_leaf = np.unique(np.logspace(np.log10(5), np.log10(300), 20, endpoint=True, dtype=int))
n_min_samples_leaf = min_samples_leaf.shape[0]
alpha = np.array([0.1, 0.3, 0.7, 0.9])
n_alphas = alpha.shape[0]
scale_normal = 0.1 # std of the normal random variable used for the additional noise

# Quantile averaged in leaf
method = qosa_Quantile__Averaged_Quantile(alpha=alpha,
                                          n_estimators=n_trees,
                                          min_samples_leaf=min_samples_leaf,
                                          used_bootstrap_samples=False,
                                          optim_by_CV=True,
                                          CV_strategy="OOB")

results = [np.empty((n_RMSE, n_alphas, BiDifferenceExponential().dim), dtype=np.float64) for i in range(1)]
for i in tqdm(range(n_RMSE)):
    X1 = rng.exponential(size = (n_samples, 2))
    Y1 = X1[:,0] - X1[:,1] + rng.normal(scale=scale_normal, size=n_samples)

    # Second sample to estimate the outer expectation of the index
    X2 = rng.exponential(size = (n_samples, 2))
    Y2 = X2[:,0] - X2[:,1] + rng.normal(scale=scale_normal, size=n_samples)
        
    qosa = QuantileBasedQosaIndices()
    qosa.feed_sample(X1, Y1, X2, Y2)
    qosa_results = qosa.compute_indices(method=method)
    results[0][i,:,:] = qosa_results.qosa_indices_estimates
    
# Percentage of noise included in the output random variable
IC_output = laplace.ppf(0.975) - laplace.ppf(0.025)
IC_noise = norm.ppf(q=0.975, scale=scale_normal) - norm.ppf(q=0.025, scale=scale_normal)
(IC_noise/IC_output)*100

# -----------------------------------------------------------------------------
# Get the true values of the QOSA indices
# -----------------------------------------------------------------------------

model.alpha = alpha
qosa_indices_theoretical = model.qosa_indices

# -----------------------------------------------------------------------------
# Plot the estimated values for each method
# -----------------------------------------------------------------------------

output_path = ('./%s_n_RMSE_%d_n_samples_%d_n_min_samples_leaf_%d_n_trees_%d.pdf' % (model.name, 
                n_RMSE, n_samples, n_min_samples_leaf, n_trees))
pdf_pages = PdfPages(output_path)

l_ax = [r'$\widehat{S}^{\alpha}$ with $\widehat{Q}^{1,o}$',
        r'$\widehat{S}^{\alpha}$ with $\widehat{Q}^{2,o}$']

 
colors = sns.color_palette("bright")
medianprops = {"linewidth":2, "color":"black"}

for i in range(1):
    y_max = max(qosa_indices_theoretical.max(), results[i].max())
    fig, axes = plt.subplots(figsize=(12,8))
    axes.set_ylim(0., y_max+0.05)
    
    # -------------------------------
    # Boxplot of the estimated values
    # -------------------------------
    for j in range(dim):
        boxplot_figure = axes.boxplot(results[i][:,:,j], medianprops=medianprops, patch_artist=True)
        
        # fill color of the boxes
        for box in boxplot_figure["boxes"]:
            # change fill color
            box.set(facecolor = colors[j])
    
        # change color and linewidth of the whiskers
        for whisker in boxplot_figure["whiskers"]:
            whisker.set(linestyle = "--")
        
        # change the style of fliers and their fill
        for flier in boxplot_figure["fliers"]:
            flier.set(markerfacecolor = colors[j])
    
    # -------------------------
    # Points of the true values
    # ------------------------- 
    axes2 = axes.twinx()
    axes2.set_ylim(axes.get_ylim())
    axes2.set_yticks([]) 
    for j in range(dim):
        axes2.plot(np.arange(n_alphas)+1, qosa_indices_theoretical[:,j], linestyle='None', marker='s', markersize=6, color=colors[4])

    # -----------------------------------
    # Customization of the axes and Title
    # -----------------------------------
    axes.set_xlabel("Values of " + r"$ \alpha $", fontsize=16)
    axes.set_xticks(np.arange(n_alphas)+1)
    axes.xaxis.set_ticklabels(alpha, fontsize=16)
    axes.set_ylabel("QOSA", fontsize=16)
    axes.yaxis.set_tick_params(labelsize=16)
    axes.set_title('Distribution of ' + l_ax[i] + ' and ' + '$N_{sample}=%d,'
                    'N_{tree}=%d, N_{min\_samples\_leaf}=%d, N_{RMSE}=%d$' 
                    % (n_samples, n_trees, n_min_samples_leaf, n_RMSE),
                    fontsize = 16)
    
    # ------
    # Legend
    # ------
    legend_elements = [Patch(facecolor=colors[j], label= r"$ S^{\alpha}_{X_{" + str(j+1) + "}} $") for j in range(dim)]
    legend_elements.append(Line2D([0], [0], marker='s', color='w', label='True \n values', markerfacecolor=colors[4], markersize=8))
    axes.legend(handles=legend_elements, fontsize=16, loc="center left", bbox_to_anchor=(1., 0.5))
        
    fig.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    
pdf_pages.close()
