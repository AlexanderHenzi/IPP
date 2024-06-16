# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages # to make several figures in one pdf
import numpy as np
import seaborn as sns


alpha = np.arange(0.01, 1, 0.01)
n_alphas = alpha.shape[0]
dim = 2


# -----------------------------------------------------------------------------
#
# Compute the QOSA indices
#
# -----------------------------------------------------------------------------

qosa_indices = np.zeros((n_alphas, dim), dtype=np.float64)
for i, alpha_temp in enumerate(alpha):
    if alpha_temp >= 0.5:
        qosa_indices[i,0] = ((1-alpha_temp)*(1-np.log(2*(1-alpha_temp)))+
                            alpha_temp*np.log(alpha_temp))/(
                            (1-alpha_temp)*(1-np.log(2*(1-alpha_temp))))
        qosa_indices[i,1] = ((1-alpha_temp)*(1-np.log(2*(1-alpha_temp)))+
                            (1-alpha_temp)*np.log(1-alpha_temp))/(
                            (1-alpha_temp)*(1-np.log(2*(1-alpha_temp))))
    else:
        qosa_indices[i,0] = (alpha_temp*(1-np.log(2*alpha_temp))+
                            alpha_temp*np.log(alpha_temp))/(
                            alpha_temp*(1-np.log(2*alpha_temp)))
        qosa_indices[i,1] = (alpha_temp*(1-np.log(2*alpha_temp))+
                            (1-alpha_temp)*np.log(1-alpha_temp))/(
                            alpha_temp*(1-np.log(2*alpha_temp)))


# -----------------------------------------------------------------------------
#
# Plot the result
#
# -----------------------------------------------------------------------------
     
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
    

set_style_paper()

pdf_pages = PdfPages('difference_expo_reference.pdf')

colors = sns.color_palette('bright')

fig, axes = plt.subplots(figsize=(8,6))

for i in range(dim):
    axes.plot(alpha, qosa_indices[:,i], linestyle='-', linewidth=3, color=colors[i], label = r"$ S_{%d}^{\alpha} $" %(i+1,))

axes.axhline(y=0.5, color=colors[2], linestyle='--', linewidth=3, label = r"$ S_{%d} = S_{%d} $" %(1,2))

# -----------------------------------
# Customization of the axes and Title
# -----------------------------------
axes.set_xlabel('Values of ' + r'$\alpha$', fontsize=20)
axes.set_ylabel('QOSA', fontsize=20)
axes.tick_params(axis = 'both', labelsize = 20)
axes.set_title('QOSA & Sobol indices', fontsize = 20)

# ------
# Legend
# ------

axes.legend(loc='best',frameon=True, fontsize = 20)
   
fig.tight_layout()
pdf_pages.savefig(fig, bbox_inches='tight')
plt.close(fig)
pdf_pages.close()



# -----------------------------------------------------------------------------
#
# Plot the density
#
# -----------------------------------------------------------------------------

import scipy.stats as ss

color = sns.color_palette('bright')
fig, axes = plt.subplots(figsize = (6,6))
x = np.linspace(0,5,5000)

# Plot with some specific colors for the QOSA indices
axes.plot(x,ss.expon.pdf(x), 'r-', lw=3, label=r'$X_{1}: \mathcal{E}(1)$', color = color[0])
axes.plot(-x,ss.expon.pdf(x), 'b-', lw=3, label=r'$X_{2}: -\mathcal{E}(1)$', color = color[1])

axes.tick_params(axis = 'both', labelsize = 20)
axes.set_xlabel(r'$x$', fontsize = 20)
axes.set_ylabel(r'$f(x)$', fontsize = 20)
axes.set_title('Density of the inputs', fontsize = 20)
axes.legend(loc='upper left',frameon = True, fontsize = 20)

fig.tight_layout()
fig.savefig('density_inputs.pdf', bbox_inches='tight')