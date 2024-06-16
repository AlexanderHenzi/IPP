# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
#plt.switch_backend('Agg') # very important to plot on cluster
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

from qosa import QuantileBasedQosaIndices, Kernel_CDF
from qosa.plots import set_style_paper
from qosa.tests import DifferenceExponential

savefig = True

# -------------------------------
# |||||||||||||||||||||||||||||||
# -------------------------------
#
# Settings to compute the indices
#
# -------------------------------
# |||||||||||||||||||||||||||||||
# -------------------------------

lambda_param = 1
model = DifferenceExponential(lambda_param)
dim = model.dim

n_RMSE = 100
n_samples = 4*10**4
alpha = np.array([0.01, 0.3, 0.7, 0.99])
n_alphas = alpha.shape[0]
method = Kernel_CDF(alpha=alpha,
                    bandwidth=None,
                    optim_by_CV=False)

# ----
# ||||
# ----
#
# Loop
#
# ----
# ||||
# ----

model.alpha = alpha
QOSA_indices_theoretical = model.qosa_indices
qosa = QuantileBasedQosaIndices()

QOSA_indices = np.zeros((n_alphas, dim, n_RMSE), dtype=np.float64)

for i in range(n_RMSE):
    print(i)
    qosa.build_sample(model=model, n_samples=n_samples)
    qosa_results = qosa.compute_indices(method)

    QOSA_indices[:, :, i] = qosa_results.qosa_indices_estimates

# ----------------
# ||||||||||||||||
# ----------------
#
# Plot the results
#
# ----------------
# ||||||||||||||||
# ----------------

set_style_paper()

colors = sns.color_palette('bright')
medianprops = dict(linewidth=2, color='black')

fig, axes = plt.subplots(figsize=(12,8))
axes.set_ylim(-0.25, 1.25)

# -------------------------------
# Boxplot of the estimated values
# -------------------------------

for i in range(dim):
    boxplot_figure = axes.boxplot(QOSA_indices[:,i,:].T, medianprops=medianprops, patch_artist=True)
    
    # fill color of the boxes
    for box in boxplot_figure["boxes"]:
        # change fill color
        box.set(facecolor = colors[i])

    # change color and linewidth of the whiskers
    for whisker in boxplot_figure["whiskers"]:
        whisker.set(linestyle = '--')
    
    # change the style of fliers and their fill
    for flier in boxplot_figure["fliers"]:
        flier.set(markerfacecolor = colors[i])
            
# -------------------------
# Points of the true values
# ------------------------- 

if QOSA_indices_theoretical is not None:
    axes2 = axes.twinx()
    axes2.set_ylim(axes.get_ylim())
    axes2.set_yticks([]) 
    for i in range(dim):
        axes2.plot(np.arange(n_alphas)+1, QOSA_indices_theoretical[:,i], linestyle='None', marker='s', markersize=6, color=colors[i])

# -----------------------------------
# Customization of the axes and Title
# -----------------------------------

axes.set_xlabel('Values of ' + r'$\alpha$', fontsize=16)
axes.xaxis.set_ticklabels(alpha, fontsize=16)
axes.set_ylabel('QOSA', fontsize=16)
axes.yaxis.set_tick_params(labelsize=16)
axes.set_title('Distribution of ' + r'$S^{\alpha}$' + ' with ' + '$N_{sample}=%d,'
                'bandwidth = n^{-0.2}, N_{RMSE}=%d$' 
                % (n_samples, n_RMSE), fontsize = 16)

# ------
# Legend
# ------
legend_elements = [Patch(facecolor=colors[i], label=r'$S^{\alpha}_{X_{%d}}$' % (i+1,)) for i in range(dim)]
if QOSA_indices_theoretical is not None:
    legend_elements.append(Line2D([0], [0], marker='s', color='w', label='True \n values', markerfacecolor=colors[4], markersize=8))
axes.legend(handles=legend_elements, fontsize=16, loc='center left', bbox_to_anchor=(1., 0.5))
    
fig.tight_layout()
if savefig:
    fig.savefig("model_diff_expo_Kernel_CDF_bandwidth_default_n_sample_" + str(n_samples) + 
                "_n_loop_" + str(n_RMSE) + ".pdf", bbox_inches="tight")
