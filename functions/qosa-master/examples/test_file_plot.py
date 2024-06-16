# -*- coding: utf-8 -*-

"""
Add docstring of the module
"""


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.switch_backend('Agg') # very important to plot on cluster
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages # to make several figures in one pdf

from qosa import QosaIndices


# -----------------------------------------------------------------------------
#
# Design of the Additive Exponential model
#
# -----------------------------------------------------------------------------

from qosa.tests import AdditiveExponential

lambda_params = [0.5, 1., 1.5, 2.]
model = AdditiveExponential(lambda_params=lambda_params)

alpha = np.asarray([0.05, 0.1, 0.5, 0.7, 0.99])
model.alpha = alpha


# -----------------------------------------------------------------------------
#
# Compute the qosa indices
#
# -----------------------------------------------------------------------------

def plot_conference(model,
                    alpha,
                    n_samples,
                    method,
                    n_estimators,
                    optim_by_CV,
                    n_fold,
                    min_samples_leaf_start,
                    min_samples_leaf_stop,
                    min_samples_leaf_num,
                    n_RMSE):

    n_alphas = alpha.shape[0]

    # Dimension of the model
    dim = model.dim

    # Size of the leaf nodes to try
    min_samples_leaf = np.unique(np.logspace(np.log10(min_samples_leaf_start), 
                                             np.log10(min_samples_leaf_stop),
                                             min_samples_leaf_num, 
                                             endpoint=True,
                                             dtype=int))

    # ---------------------------------
    #
    # Compute RMSE for the qosa indices
    #
    # ---------------------------------

    QOSA_indices_theoretical = model.qosa_indices
    qosa = QosaIndices(model.input_distribution)

    if optim_by_CV:
        QOSA_indices = np.zeros((n_alphas, dim, n_RMSE), dtype=np.float64)
        optim_min_samples_leaf_by_dim_and_alpha = np.zeros((n_alphas, dim, n_RMSE), 
                                                           dtype=np.float64)

        start = time.time()
        for i in range(n_RMSE):
            print(i)
            qosa.build_sample(model=model,
                              n_samples=n_samples,
                              estimator=method)
            qosa_results = qosa.compute_indices(alpha,
                                                n_estimators=n_estimators,
                                                min_samples_leaf=min_samples_leaf,
                                                optim_by_CV=optim_by_CV,
                                                n_fold=n_fold)
            QOSA_indices[:,:,i] = qosa_results.qosa_indices
            optim_min_samples_leaf_by_dim_and_alpha[:,:,i] = qosa_results.min_samples_leaf_by_dim_and_alpha
        print('Total time is:{}'.format(time.time()-start))

        # QOSA indices
        QOSA_indices_RMSE = np.zeros_like(QOSA_indices)
        QOSA_indices_RMSE_normalized = np.zeros_like(QOSA_indices)

        for i in range(n_alphas):
            for j in range(dim):
                QOSA_indices_RMSE[i,j,:] = (QOSA_indices[i,j,:]-
                                            QOSA_indices_theoretical[i,j])
                QOSA_indices_RMSE_normalized[i,j,:] = (QOSA_indices_RMSE[i,j,:]/
                                                       QOSA_indices_theoretical[i,j])
            
        QOSA_indices_RMSE = np.sqrt((QOSA_indices_RMSE**2).mean(axis=2))
        QOSA_indices_RMSE_normalized = np.sqrt((QOSA_indices_RMSE_normalized**2).mean(axis=2))

        # ----------
        # Save files
        # ----------

        savefile = ('./%s_%s_with_CV_RMSE_QOSA_indices_with_n_RMSE_%d_min_samples_leaf_num_'
                    '%d_n_samples_%d_n_trees_%d' % (model.name, method, n_RMSE, 
                    min_samples_leaf_num, n_samples, n_estimators))
        np.savez(savefile,
                 optim_min_samples_leaf_by_dim_and_alpha,
                 QOSA_indices_RMSE,
                 QOSA_indices_RMSE_normalized)
    else:
        QOSA_indices = np.zeros((n_alphas, dim, min_samples_leaf_num, n_RMSE), dtype=np.float64)

        start = time.time()
        for i in range(n_RMSE):
            print(i)
            qosa.build_sample(model=model,
                              n_samples=n_samples,
                              estimator=method)

            for j,min_samples_leaf_temp in enumerate(min_samples_leaf):
                QOSA_indices[:,:,j,i] = qosa.compute_indices(
                                                    alpha,
                                                    n_estimators=n_estimators,
                                                    min_samples_leaf=min_samples_leaf_temp,
                                                    optim_by_CV=optim_by_CV).qosa_indices
        print('Total time is:{}'.format(time.time()-start))

        # QOSA indices
        QOSA_indices_RMSE = np.zeros_like(QOSA_indices)
        QOSA_indices_RMSE_normalized = np.zeros_like(QOSA_indices)

        for i in range(n_alphas):
            for j in range(dim):
                QOSA_indices_RMSE[i,j,:,:] = (QOSA_indices[i,j,:,:]-
                                              QOSA_indices_theoretical[i,j])
                QOSA_indices_RMSE_normalized[i,j,:,:] = (QOSA_indices_RMSE[i,j,:,:]/
                                                         QOSA_indices_theoretical[i,j])
    
        QOSA_indices_RMSE = np.sqrt((QOSA_indices_RMSE**2).mean(axis=3))
        QOSA_indices_RMSE_normalized = np.sqrt((QOSA_indices_RMSE_normalized**2).mean(axis=3))

        # ----------
        # Save files
        # ----------

        savefile = ('./%s_%s_without_CV_RMSE_QOSA_indices_with_n_RMSE_%d_min_samples_leaf_num_'
                    '%d_n_samples_%d_n_trees_%d' % (model.name, method, n_RMSE,
                    min_samples_leaf_num, n_samples, n_estimators))
        np.savez(savefile,
                 QOSA_indices_RMSE,
                 QOSA_indices_RMSE_normalized)
        
    return QOSA_indices, QOSA_indices_RMSE, QOSA_indices_RMSE_normalized

    # -----------------------------
    #
    # Plot the results for the RMSE
    #
    # -----------------------------

#    pdf_pages = PdfPages(savefile+'.pdf')
#
#    if optim_by_CV:
#
#        #----------------------------
#        # classic and normalized RMSE
#        #----------------------------
#
#        index_names = [r'$\alpha = %.2f$' % (alpha[i],) for i in range(n_alphas)]
#
#        for i in range(dim):
#            # First version
#            fig, axes = plt.subplots(figsize = (12,10))
#            axes.scatter(alpha, QOSA_indices_RMSE[:,i], marker='o', s=60)
#            axes.scatter(alpha, QOSA_indices_RMSE_normalized[:,i], marker='o', s=60)
#            axes.set_xlabel('Values of ' + r'$\alpha$', fontsize=16)
#            axes.xaxis.set_tick_params(labelsize=16)
#            axes.yaxis.set_tick_params(labelsize=16)
#            axes.legend(['RMSE', 'Normalized RMSE'], fontsize=16)
#            axes.set_title(r'$S^{\alpha}_{X_%d}$' %(i+1,) + ' with ' + '$N_{sample}=%d,' 
#                            ' N_{tree}=%d, N_{min\_samples\_leaf}=%d, N_{RMSE}=%d$' 
#                            % (n_samples, n_estimators, min_samples_leaf_num,
#                               n_RMSE), fontsize = 16)
#            fig.tight_layout()
#            pdf_pages.savefig(fig, bbox_inches='tight')
#            plt.close(fig)
#
#            # Second version, usinb Bar plot
#            fig, axes = plt.subplots(figsize = (12,10))
#            df_RMSE_values = pd.DataFrame(np.c_[QOSA_indices_RMSE[:,i], QOSA_indices_RMSE_normalized[:,i]],
#                              columns=['RMSE', 'Normalized RMSE'],
#                              index=index_names)
#            df_RMSE_values.plot.bar(ax=axes, rot=0, fontsize=16)
#            axes.set_xlabel('Values of ' + r'$\alpha$', fontsize=16)
#            axes.xaxis.set_tick_params(labelsize=16)
#            axes.yaxis.set_tick_params(labelsize=16)
#            axes.set_title(r'$S^{\alpha}_{X_%d}$' %(i+1,) + ' with ' + '$N_{sample}=%d,' 
#                            ' N_{tree}=%d, N_{min\_samples\_leaf}=%d, N_{RMSE}=%d$' 
#                            % (n_samples, n_estimators, min_samples_leaf_num,
#                               n_RMSE), fontsize = 16)
#            axes.legend(fontsize = 16)
#            fig.tight_layout()
#            pdf_pages.savefig(fig, bbox_inches='tight')
#            plt.close(fig)
#
#        #--------------------------------------------
#        # Boxplot and ViolinPlot for the qosa indices
#        #--------------------------------------------
#
#        Scor_color_1 = (0, 0.4196078431, 0.5529411764)
#        Scor_color_2 = (0.4980392156, 0.4980392156, 0.4980392156)
#        Scor_color_3 = (0.4901960784, 0.7882352941, 0.9686274509)
#
#        meanpointprops = dict(marker='D', 
#                              markerfacecolor=Scor_color_1,
#                              markeredgecolor='black',
#                              markersize=12)
#        columns_names = [r'$\alpha = %.2f$' % (alpha[i],) for i in range(n_alphas)]
#        
#        for i in range(dim):
#            # First version of Boxplot
#            fig, axes = plt.subplots(figsize=(12,10))
#            axes.boxplot(QOSA_indices[:,i,:].T, showmeans=True)
#            axes.scatter(np.arange(n_alphas)+1,QOSA_indices_theoretical[:,i], marker='o', s=60)
#            axes.set_xlabel('Values of ' + r'$\alpha$', fontsize=16)
#            axes.xaxis.set_ticklabels(alpha, fontsize=16)
#            axes.yaxis.set_tick_params(labelsize=16)
#            axes.set_title(r'$S^{\alpha}_{X_%d}$' %(i+1,) + ' with ' + '$N_{sample}=%d,' 
#                            ' N_{tree}=%d, N_{min\_samples\_leaf}=%d, N_{RMSE}=%d$' 
#                            % (n_samples, n_estimators, min_samples_leaf_num,
#                               n_RMSE), fontsize = 16)
#            fig.tight_layout()
#            pdf_pages.savefig(fig, bbox_inches='tight')
#            plt.close(fig)
#
#            # Second version of Boxplot
#            fig, axes = plt.subplots(figsize=(12,10))
#            df_qosa_indices = pd.DataFrame(QOSA_indices[:,i,:].T, columns=columns_names)
#            #sns.boxplot(data=df_qosa_indices, ax=axes, palette='bright', showmeans=True, meanprops=meanpointprops)
#            sns.boxplot(data=df_qosa_indices, ax=axes, color=Scor_color_3, showmeans=True, meanprops=meanpointprops)
#            sns.scatterplot(x=np.arange(n_alphas), y=QOSA_indices_theoretical[:,i], ax=axes, markers='o', color=Scor_color_2, edgecolor='black', s=150)
#            axes.set_xlabel('Values of ' + r'$\alpha$', fontsize=16)
#            axes.xaxis.set_tick_params(labelsize=16)
#            axes.yaxis.set_tick_params(labelsize=16)
#            axes.set_title(r'$S^{\alpha}_{X_%d}$' %(i+1,) + ' with ' + '$N_{sample}=%d,' 
#                            ' N_{tree}=%d, N_{min\_samples\_leaf}=%d, N_{RMSE}=%d$' 
#                            % (n_samples, n_estimators, min_samples_leaf_num,
#                               n_RMSE), fontsize = 16)
#            legend_elements = [Line2D([0], [0], marker='D', color='w', label='Mean', markerfacecolor=Scor_color_1, markeredgecolor='black', markersize=12),
#                               Line2D([0], [0], marker='o', color='w', label='True value', markerfacecolor=Scor_color_2, markeredgecolor='black', markersize=12)]
#            axes.legend(handles=legend_elements, fontsize=16, loc='best')
#            fig.tight_layout()
#            pdf_pages.savefig(fig, bbox_inches='tight')
#            plt.close(fig)
#
#            # ViolinPlot
#            fig, axes = plt.subplots(figsize=(12,10))
#            df_qosa_indices = pd.DataFrame(QOSA_indices[:,i,:].T, columns = columns_names)
#            sns.violinplot(data=df_qosa_indices, ax=axes, color=Scor_color_3)
#            sns.scatterplot(x=np.arange(n_alphas), y=QOSA_indices_theoretical[:,i], ax=axes, marker='D', color=Scor_color_1, edgecolor='black', s=120)
#            axes.set_xlabel('Values of ' + r'$\alpha$', fontsize=16)
#            axes.xaxis.set_tick_params(labelsize=16)
#            axes.yaxis.set_tick_params(labelsize=16)
#            axes.set_title(r'$S^{\alpha}_{X_%d}$' %(i+1,) + ' with ' + '$N_{sample}=%d,' 
#                            ' N_{tree}=%d, N_{min\_samples\_leaf}=%d, N_{RMSE}=%d$' 
#                            % (n_samples, n_estimators, min_samples_leaf_num,
#                               n_RMSE), fontsize = 16)
#            legend_elements = [Line2D([0], [0], marker='D', color='w', label='True value', markerfacecolor=Scor_color_1, markeredgecolor='black', markersize=10)]
#            axes.legend(handles=legend_elements, fontsize=16, loc='best')
#            fig.tight_layout()
#            pdf_pages.savefig(fig, bbox_inches='tight')
#            plt.close(fig)
#
#        #---------------------------------------------------------------------------
#        # Boxplot and ViolinPlot for the optimized min_samples_leaf by dim and alpha
#        #---------------------------------------------------------------------------
#            
#        for i in range(dim):
#            # Boxplot
#            fig, axes = plt.subplots(figsize=(12,10))
#            axes.boxplot(optim_min_samples_leaf_by_dim_and_alpha[:,i,:].T, showmeans=True)
#            axes.set_xlabel('Values of ' + r'$\alpha$', fontsize=16)
#            axes.xaxis.set_ticklabels(alpha, fontsize=16)
#            axes.yaxis.set_tick_params(labelsize=16)
#            axes.set_title(r'Distribution of ' + '$min\_samples\_leaf$' + ' for ' +
#                            r'$S^{\alpha}_{X_%d}$' %(i+1,) + ' with ' + 
#                            '$N_{min\_samples\_leaf}=%d, N_{RMSE}=%d$' 
#                            % (min_samples_leaf_num, n_RMSE), fontsize = 16)
#            fig.tight_layout()
#            pdf_pages.savefig(fig, bbox_inches='tight')
#            plt.close(fig)
#
#            # ViolinPlot
#            fig, axes = plt.subplots(figsize=(12,10))
#            df_min_samples_leaf_by_dim_and_alpha = pd.DataFrame(optim_min_samples_leaf_by_dim_and_alpha[:,i,:].T, columns = columns_names)
#            sns.violinplot(data=df_min_samples_leaf_by_dim_and_alpha, ax=axes, color=Scor_color_3)
#            axes.set_xlabel('Values of ' + r'$\alpha$', fontsize=16)
#            axes.xaxis.set_tick_params(labelsize=16)
#            axes.yaxis.set_tick_params(labelsize=16)
#            axes.set_title(r'Distribution of ' + '$min\_samples\_leaf$' + ' for ' +
#                            r'$S^{\alpha}_{X_%d}$' %(i+1,) + ' with ' + 
#                            '$N_{min\_samples\_leaf}=%d, N_{RMSE}=%d$' 
#                            % (min_samples_leaf_num, n_RMSE), fontsize = 16)
#            fig.tight_layout()
#            pdf_pages.savefig(fig, bbox_inches='tight')
#            plt.close(fig)
#    else:
#
#        #----------------------------
#        # classic and normalized RMSE
#        #----------------------------
#        
#        for i, alpha_temp in enumerate(alpha):
#            for j in range(dim):
#                fig, axes = plt.subplots(figsize = (14,8))
#                axes.plot(min_samples_leaf, QOSA_indices_RMSE[i,j,:], marker='o', markersize=8, linewidth=2)
#                axes.plot(min_samples_leaf, QOSA_indices_RMSE_normalized[i,j,:], marker='o', markersize=8, linewidth=2)
#                axes.set_xlabel('Values of min_samples_leaf', fontsize=16)
#                axes.xaxis.set_tick_params(labelsize=16)
#                axes.yaxis.set_tick_params(labelsize=16)
#                axes.legend(['RMSE', 'Normalized RMSE'], fontsize=16)
#                axes.set_title(r'$S^{\alpha = %.2f}_{X_%d}$' %(alpha_temp, j+1) +
#                                ' with ' + '$N_{sample}=%d, N_{tree}=%d, N_{RMSE}=%d$' 
#                                % (n_samples, n_estimators, n_RMSE), fontsize=16)
#                fig.tight_layout()
#                pdf_pages.savefig(fig, bbox_inches='tight')
#                plt.close(fig) 
#                
#        #-----------------------------
#        # Boxplot for the qosa indices
#        #-----------------------------
#            
#        for i,alpha_temp in enumerate(alpha):
#            for j in range(dim):      
#                fig, axes = plt.subplots(figsize=(14,8))
#                axes.boxplot(QOSA_indices[i,j,:,:].T, showmeans=True)
#                axes.axhline(QOSA_indices_theoretical[i,j], linestyle='--')
#                axes.set_xlabel('Values of min_samples_leaf', fontsize=16)
#                axes.xaxis.set_ticklabels(min_samples_leaf, fontsize=16, rotation=60)
#                axes.yaxis.set_tick_params(labelsize = 16)
#                axes.set_title(r'$S^{\alpha = %.2f}_{X_%d}$' %(alpha_temp, j+1) +
#                                ' with ' + '$N_{sample}=%d, N_{tree}=%d, N_{RMSE}=%d$' 
#                                % (n_samples, n_estimators, n_RMSE), fontsize=16)
#                fig.tight_layout()
#                pdf_pages.savefig(fig, bbox_inches='tight')
#                plt.close(fig)
#
#    pdf_pages.close()
    
    
parameters = {'model': model,
              'alpha': alpha,
              'n_samples': 3*10**3,
              'method': 'Forest_1',
              'n_estimators': 10**1,
              'optim_by_CV': True,
              'n_fold': 3,
              'min_samples_leaf_start': 5,
              'min_samples_leaf_stop': 1500,
              'min_samples_leaf_num': 10,
              'n_RMSE': 10}
QOSA_indices, QOSA_indices_RMSE, QOSA_indices_RMSE_normalized = plot_conference(**parameters)

dim=model.dim
n_alphas = alpha.size
QOSA_indices_theoretical = model.qosa_indices
n_samples = 10**3
n_estimators = 10
min_samples_leaf_num = 5
n_RMSE=10

colors = sns.color_palette('bright')
medianprops = dict(linewidth=2, color='black')

# -----------------------------------------------------------------------------
#
# QOSA INDICES
#
# -----------------------------------------------------------------------------
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
        flier.set(markerfacecolor=colors[i])
            
# -------------------------
# Points of the true values
# -------------------------  
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
                'N_{tree}=%d, N_{min\_samples\_leaf}=%d, N_{RMSE}=%d$' 
                % (n_samples, n_estimators, min_samples_leaf_num,
                   n_RMSE), fontsize = 16)

# ------
# Legend
# ------
legend_elements = [Patch(facecolor=colors[i], label=r'$S^{\alpha}_{X_{%d}}$' % (i+1,)) for i in range(dim)]
legend_elements.append(Line2D([0], [0], marker='s', color='w', label='True \n values', markerfacecolor=colors[4], markersize=8))
axes.legend(handles=legend_elements, fontsize=16, loc='center left', bbox_to_anchor=(1., 0.5))
    
fig.tight_layout()



# -----------------------------------------------------------------------------
#
# RMSE
#
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(figsize=(12,8))

# -------------------------------------------
# Points of the classical and normalized RMSE
# -------------------------------------------
for i in range(dim):
    axes.plot(range(n_alphas), QOSA_indices_RMSE[:,i], linestyle='None', marker=5, markersize=15, color=colors[i])

axes2 = axes.twinx()
for i in range(dim):
    axes2.plot(range(n_alphas), QOSA_indices_RMSE_normalized[:,i], linestyle='None', marker=4, markersize=15, color=colors[i])
    
# -----------------------------------
# Customization of the axes and Title
# -----------------------------------
axes.set_xticks(range(n_alphas))
axes.xaxis.set_ticklabels(alpha, fontsize=16)
axes.set_xlabel('Values of ' + r'$\alpha$', fontsize=16)

axes.yaxis.set_tick_params(labelsize=16)
axes2.yaxis.set_tick_params(labelsize=16)
axes.set_title(r'$N_{sample}=%d, N_{tree}=%d, N_{min\_samples\_leaf}=%d, N_{RMSE}=%d$' 
                % (n_samples, n_estimators, min_samples_leaf_num, n_RMSE),
                fontsize = 16)
# ------
# Legend
# ------
legend_elements_RMSE = [Line2D([0], [0], 
                               marker=5,
                               color='w',
                               label='RMSE',
                               markerfacecolor= 'black',
                               markersize=15),
                        Line2D([0], [0], 
                               marker=4,
                               color='w',
                               label='Normalized \n RMSE',
                               markerfacecolor='black',
                               markersize=15),]
legend_elements = [Patch(facecolor=colors[i], label=r'$X_{%d}$' % (i+1,)) for i in range(dim)]
legend = legend_elements_RMSE + legend_elements
axes.legend(handles=legend, fontsize=16, loc='center left', bbox_to_anchor=(1.05, 0.5))
        
fig.tight_layout()
