# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
#plt.switch_backend('Agg') # very important to plot on cluster
from matplotlib.patches import Patch
import numba as nb
import numpy as np
import openturns as ot
import seaborn as sns

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

n_samples=  10**5
dim = 3
GPD_params=[1.5, 0.4]
LN_params=[1., 0.7, 0.]
Gamma_params=[2.8, 0.7, 0.]
alpha = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
n_alphas = alpha.shape[0]
n_loop = 10**2

margins = [ot.GeneralizedPareto(*GPD_params), ot.LogNormal(*LN_params), ot.Gamma(*Gamma_params)]
copula = ot.IndependentCopula(dim)
input_distribution = ot.ComposedDistribution(margins, copula)


# ---------
# |||||||||
# ---------
#
# Functions
#
# ---------
# |||||||||
# ---------

def set_style_paper(context='paper'):
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context(context)
    
    # Set the font to be serif, rather than sans
    sns.set(font='serif')
    
    # Make the background white, and specify the
    # specific font family
    sns.set_style('white', {
        'font.family': 'serif',
        'font.serif': ['Times', 'Palatino', 'serif']
    })

    
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


# ----
# ||||
# ----
#
# Loop
#
# ----
# ||||
# ----
    
QOSA_indices = np.zeros((n_loop, dim, n_alphas), dtype=np.float64)
numerator = np.zeros((dim, n_alphas), dtype=np.float64)
for i in range(n_loop):
    input_sample = np.asarray(input_distribution.getSample(n_samples))
    output_sample = input_sample.sum(axis=1)
        
    # Compute the denominator of the indices
    alpha_quantile = np.percentile(output_sample, q=alpha*100)
    denominator = _averaged_check_function_alpha_array(
                                                output_sample.reshape(-1,1)-alpha_quantile,
                                                alpha)
    
    for j in range(dim):
        conditional_quantile = np.percentile(np.delete(input_sample, j, axis=1).sum(axis=1), q=alpha*100)
        conditional_quantile = input_sample[:,j].reshape(-1,1) + conditional_quantile
        numerator[j,:] = _averaged_check_function_alpha_array(
                                                output_sample.reshape(-1,1)-conditional_quantile,
                                                alpha)
    
    QOSA_indices[i,:,:] = 1 - numerator/denominator


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

colors = sns.color_palette("bright")
medianprops = {"linewidth":2, "color":"black"}

fig, axes = plt.subplots(figsize=(12,8))
# -------------------------------
# Boxplot of the estimated values
# -------------------------------
for i in range(dim):
    boxplot_figure = axes.boxplot(QOSA_indices[:,i,:], medianprops=medianprops, patch_artist=True)
    
    # fill color of the boxes
    for box in boxplot_figure["boxes"]:
        # change fill color
        box.set(facecolor = colors[i])

    # change color and linewidth of the whiskers
    for whisker in boxplot_figure["whiskers"]:
        whisker.set(linestyle = "--")
    
    # change the style of fliers and their fill
    for flier in boxplot_figure["fliers"]:
        flier.set(markerfacecolor = colors[i])

# -----------------------------------
# Customization of the axes and Title
# -----------------------------------
axes.set_xlabel("Values of " + r"$ \alpha $", fontsize=16)
axes.xaxis.set_ticklabels(alpha, fontsize=16)
axes.set_ylabel("QOSA", fontsize=16)
axes.yaxis.set_tick_params(labelsize=16)
axes.set_title("Distribution of " + r"$ S^{\alpha} $" + " with " +
               r"$ N_{sample}= $" + str(n_samples) + r", $ N_{loop}= $" + str(n_loop),
               fontsize = 16)

# ------
# Legend
# ------
legend_elements = [Patch(facecolor=colors[i], label= r"$ S^{\alpha}_{X_{" + str(i+1) + "}} $") for i in range(dim)]
axes.legend(handles=legend_elements, fontsize=16, loc="center left", bbox_to_anchor=(1., 0.5))
    
fig.tight_layout()
if savefig:
    fig.savefig("model_Toy_insurance_reference_n_sample_" + str(n_samples) + 
                "_n_loop_" + str(n_loop) + ".pdf", bbox_inches="tight")
