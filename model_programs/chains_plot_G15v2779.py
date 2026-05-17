"""
Program to graph the convergence of the chains for each of the gravitational lens and source parameters.
"""

import numpy as np
import matplotlib.pyplot as pl; pl.ioff()
import matplotlib; matplotlib.use('Agg')
import sys
import os
sys.path.append(os.path.abspath('..')) #Point to visilens from the examples directory
import visilens as vl
import time
import pickle
import gzip

# We'll use this to help name the output plots.
plotfbase = 'G15v2779_modelcont_3005003000'
#load the data
mcmcresult = pickle.load(gzip.open('chains_'+plotfbase+'.pzip'))

data = vl.read_visdata('G15v2.779_g15.bin')
allcols = list(mcmcresult['chains'].dtype.names)
# Gets rid of mag for unlensed sources, which is always 1.
allcols = [col for col in allcols if not ('mu' in col and np.allclose(mcmcresult['chains'][col],1.))]
print(allcols) #show the list of parameters
print(mcmcresult['chains'].shape)
print(mcmcresult.keys())
######################################################################################
#Graph all the plots in the same figure

#Number of walkers and steps
nwalkers=300
nsteps=3000

#Extract values from the chain
chains_values = np.array([list(chain) for chain in mcmcresult['chains']])
chains_reshaped = chains_values.reshape(nwalkers, nsteps, -1)

# Graphing time evolution
fig,axes=pl.subplots(len(allcols),1,figsize=(12,len(allcols) * 2),sharex=True)

for i, var in enumerate(allcols):
    ax =axes[i]
    var_index=i  #Parameter's index in the chain
    for j in range(nwalkers):
        ax.plot(np.arange(nsteps), chains_reshaped[j, :,var_index], alpha=0.1, color='b')
    #Add the median value
    median_chain = np.median(chains_reshaped[:, :, var_index], axis=0)
    if var_index == 0:
       ax.plot(np.arange(nsteps), median_chain, color='red', linewidth=2,label='Median')
       ax.legend(loc='upper right')
    else:
       ax.plot(np.arange(nsteps),median_chain,color='red',linewidth=2)
    ax.set_ylabel(var)
    ax.grid(True)
axes[-1].set_xlabel('nstep')

fig.tight_layout()
filename=f"{plotfbase}_{nwalkers}{nsteps}_allparameters.png"
fig.savefig(filename,dpi=300)
pl.close(fig)
######################################################################################
#Outlier analysis configuration
burnin = 2500 #last steps
tail = chains_reshaped[:,burnin:,:] #steady state only.
#Graph each parameter against the number of steps in different figures.
for i, var in enumerate(allcols):
    pl.figure(figsize=(10,5))
    var_index = i
    # Outlier analysis.
    param_tail = tail[:,-100:,var_index]
    
    # Median per walker.
    chain_medians = np.median(param_tail, axis=1)
    
    # 1. Find the median of the medians (the true center of the main group).
    global_median = np.median(chain_medians)
    
    # 2. Calculate how much the chains deviate from the center, in a robust way.
    mad = np.median(np.abs(chain_medians - global_median))
    
    # Prevent division by zero if all chains converge to an identical value.
    if mad == 0:
       mad = 1e-10 
        
    # 3. Definir el umbral. Multiplicar por 4.5 equivale aproximadamente a 3 desviaciones estándar.
    # You can adjust this multiplier (e.g., 3, 4 or 5) to make the filter more or less stringent.
    k = 5
    threshold = k * mad
    
    # 4. Detect outliers: any walker whose median deviates beyond the threshold.
    outliers = np.abs(chain_medians - global_median) > threshold
    # --------------------------------------------------

    n_outliers = np.sum(outliers)
    percent_outliers = 100 * n_outliers / nwalkers

    #Plot of all chains
    for j in range(nwalkers):
        if outliers[j]:
           pl.plot(np.arange(nsteps),chains_reshaped[j,:,var_index],alpha=0.4,color='orange')
        else:
            pl.plot(np.arange(nsteps),chains_reshaped[j,:,var_index],alpha=0.1,color='b')
    #Overall median
    median_chain = np.median(chains_reshaped[:, :, var_index], axis=0)
    pl.plot(np.arange(nsteps), median_chain, color='red', linewidth=2, label='Median')

    pl.text(
        0.02, 0.95,
        f'Outliers: {percent_outliers:.1f}%({n_outliers})',
        transform=pl.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    pl.plot([],[],color='b',label='Normal walkers')
    pl.plot([],[],color='orange',label='Outliers')

    pl.xlabel('nstep')
    pl.ylabel(var)
    pl.title(f'Evolution of {var}')
    pl.legend(loc='lower left')
    pl.grid(True)

    filename= f"chainsplot_modelcont_{nwalkers}{nsteps}_{var}.png"
    pl.savefig(filename, dpi=300)
    pl.close()


