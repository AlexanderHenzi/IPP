# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages # to make several figures in one pdf
import os


path = r"C:\Users\u005032\Folders\qosa_indices\qosa\examples\output"
file_name = "Toy_Insurance_Forest_1_with_CV_RMSE_QOSA_indices_with_n_RMSE_10_min_samples_leaf_num_20_n_samples_100000_n_trees_300"
file_input_path = os.path.join(path, file_name + '.npz')
file_output_path = os.path.join(path, file_name + '_NEW.pdf')


# --------------------------------------------------
# --------------------------------------------------
#
# For the toy insurance model
#
# --------------------------------------------------
# --------------------------------------------------

npzfile  = np.load(file_input_path)
npzfile.files

QOSA_indices = npzfile['arr_0']
optim_min_samples_leaf_by_dim_and_alpha = npzfile['arr_1']
QOSA_indices_RMSE = npzfile['arr_2']
QOSA_indices_RMSE_normalized = npzfile['arr_3']

from qosa.tests import ToyInsurance

model = ToyInsurance()

alpha = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
model.alpha = alpha
QOSA_indices_theoretical = model.qosa_indices

n_alphas = alpha.shape[0]
dim = model.dim
n_samples = 10**5
n_estimators = 3*10**2
min_samples_leaf_num = 20
n_fold = 5
n_RMSE = 10


# --------------------------------------------------
# --------------------------------------------------
#
# Boxplot of all the estimated indices on one figure 
#
# --------------------------------------------------
# --------------------------------------------------


from qosa.plots import set_style_paper

set_style_paper()

pdf_pages = PdfPages(file_output_path)

if dim<=10:
    colors = sns.color_palette('bright')
else:
    colors = sns.color_palette(n_colors=dim)
medianprops = dict(linewidth=2, color='black')

fig, axes = plt.subplots(figsize=(12,8))

# -------------------------------
# Boxplot of the estimated values
# -------------------------------
for i in range(dim):
    boxplot_figure = axes.boxplot(QOSA_indices[:,i,:].T, medianprops=medianprops, patch_artist=True)
    
    #  fill color of the boxes
    for box in boxplot_figure["boxes"]:
        # change fill color
        box.set(facecolor = colors[i])

    ## change color and linewidth of the whiskers
    for whisker in boxplot_figure["whiskers"]:
        whisker.set(linestyle = '--')
    
    ## change the style of fliers and their fill
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
        axes2.plot(np.arange(n_alphas)+1, QOSA_indices_theoretical[:,i], linestyle='None', marker='s', markersize=6, color=colors[4])

# -----------------------------------
# Customization of the axes and Title
# -----------------------------------
axes.set_xlabel('Values of ' + r'$\alpha$', fontsize=16)
axes.xaxis.set_ticklabels(alpha, fontsize=16)
axes.set_ylabel('QOSA', fontsize=16)
axes.yaxis.set_tick_params(labelsize=16)
axes.set_title('Distribution of ' + r'$S^{\alpha}$' + ' with ' + '$N_{sample}=%d,'
                'N_{tree}=%d, N_{min\_samples\_leaf}=%d, N_{fold}=%d, N_{RMSE}=%d$' 
                % (n_samples, n_estimators, min_samples_leaf_num, n_fold,
                   n_RMSE), fontsize = 16)

# ------
# Legend
# ------
legend_elements = [Patch(facecolor=colors[i], label=r'$S^{\alpha}_{X_{%d}}$' % (i+1,)) for i in range(dim)]
if QOSA_indices_theoretical is not None:
    legend_elements.append(Line2D([0], [0], marker='s', color='w', label='True \n values', markerfacecolor=colors[4], markersize=8))
axes.legend(handles=legend_elements, fontsize=16, loc='center left', bbox_to_anchor=(1., 0.5))
    
fig.tight_layout()
pdf_pages.savefig(fig, bbox_inches='tight')
plt.close(fig)
pdf_pages.close()