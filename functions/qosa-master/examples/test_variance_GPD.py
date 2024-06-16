import numpy as np
import openturns as ot

n_samples_array = [10**i for i in [6, 7, 8, 9]]
n_samples_label = [r'$ 10^{%d} $' % (i,) for i in [6, 7, 8, 9]]
Xi_array = [0.1, 0.2, 0.3, 0.4]
n_loop = 10**2
res = np.empty((len(Xi_array), len(n_samples_array), n_loop), dtype=np.float64)

def variance_GPD(a, b):
    return a**2/((1 - 2*b)*(1 - b)**2)

for i, Xi in enumerate(Xi_array):
    print(i)
    GPD_params = [1.5, Xi]
    distrib = ot.GeneralizedPareto(*GPD_params)
    for j, n_samples in enumerate(n_samples_array):
        for k in range(n_loop):
            res[i,j,k] = np.var(distrib.getSample(n_samples))
            
import matplotlib.pyplot as plt
plt.switch_backend('Agg') # very important to plot on cluster
from matplotlib.backends.backend_pdf import PdfPages # to make several figures in one pdf
import seaborn as sns
from qosa.plots import set_style_paper

colors = sns.color_palette('bright')
set_style_paper()
# The rcParams 'xtick.bottom' and 'ytick.left' can be used to set the ticks on or off
# because the seaborn "white" style deactivate the tick marks
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

pdf_pages = PdfPages('test_variance_GPD.pdf')

for i in range(len(Xi_array)):
    fig, axes = plt.subplots(figsize=(12, 8))
    axes.boxplot(res[i,:,:].T, patch_artist=True)
    
    axes2 = axes.twinx()
    axes2.set_ylim(axes.get_ylim())
    axes2.set_yticks([]) 
    axes2.axhline(variance_GPD(1.5, Xi_array[i]), lw=1, linestyle='--', label='True value', color=colors[2])
    axes2.legend(loc='best', frameon=True, fontsize = 14)
    
    axes.set_xlabel('n_samples', fontsize=14)
    axes.xaxis.set_ticklabels(n_samples_label, fontsize=14, rotation=60)
    axes.tick_params(axis = 'both', labelsize = 14)
    axes.set_title(r'Variance of GPD $ \left( \sigma = %.1f, \  \xi = %.1f'
                    r'\right)$ function of the sample size with n_loop = %d' 
                    % (1.5, Xi_array[i], n_loop))
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

pdf_pages.close()