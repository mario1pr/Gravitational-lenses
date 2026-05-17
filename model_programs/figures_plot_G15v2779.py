import numpy as np
import matplotlib.pyplot as pl; pl.ioff()
import matplotlib; matplotlib.use('Agg')
import pickle
import gzip
import sys
import os
sys.path.append(os.path.abspath('..')) #points to ~/visilens-master/visilens
print("Current directory:", os.getcwd())
print("sys.path:", sys.path)
import visilens as vl

# We'll use this to help name the output plots.
plotfbase = 'G15v2779_modelcont'
nwalkers,nburn,nstep = 300,500,3000
laststep = 1000 #Number of last steps taken into account for the triangle plot defined in visilens.

# Specify the field and grid parameters.
xmax = 12.5
highresbox = [-3.0, +1.5, -3.0, +2.5] # xmin, xmax, ymin, ymax
fieldres, emitres = 0.05, 0.025

# Enter the exact name of your .bin file here.
bin_archive = 'G15v2.779_g15.bin'

with open(bin_archive, 'rb') as f:
    raw_data = np.fromfile(f)
    # The last saved value is the Primary Beam.
    pb = raw_data[-1]
    # The rest is the 7-row matrix (u, v, real, imag, sigma, ant1, ant2).
    data_matrix = raw_data[:-1].reshape((7, -1))

# The row at index 4 corresponds to visdata['sigma'].
mis_sigmas = data_matrix[4, :]

print(f"Average sigma value: {np.mean(mis_sigmas):.4e}")
print(f"Maximum sigma: {np.min(mis_sigmas):.4e}")
print(f"Minimum sigma: {np.max(mis_sigmas):.4e}")

# Read the data.
data = np.fromfile(bin_archive)

# Number of columns (7: u,v,real,imag,sigma,ant1,ant2).
ncols = 7
nrows = data.size // ncols
data = data[:nrows*ncols].reshape((ncols, nrows))
sigma = data[4]  # fifth row

# Calculate Data-Model rms.
rms = ((sigma**-2).sum())**-0.5
print("rms/beam (Jy):", rms)
print("rms/beam (mJy):", rms*1e3)

print("Loading chains...")
filename = f"chains_{plotfbase}_{nwalkers}{nburn}{nstep}.pzip"
mcmcresult = pickle.load(gzip.open(filename,'rb'))

data= vl.read_visdata(bin_archive)

print("Generating triangle plot...")
f,axesarray = vl.TrianglePlot_MCMC(
    mcmcresult,
    plotmag=True,
    plotnuisance=True)

filename = f"{plotfbase}_{nwalkers}{nburn}{nstep}_triangle_{laststep}.png"
f.savefig(filename)
pl.close()
print("Done. Saved as", filename)

print("Generating panels plot...")
f,axarr = vl.plot_images(data,mcmcresult,imsize=300,pixsize=0.07,
      limits=[-4,+2,-4,+4],mapcontours=np.array([-25,-5,5,25,45,65,85,105,125]))
axarr[0][0].text(0.,-0.3,"Data contours: steps of 20$\sigma$ starting at $\pm$5; "\
            "Residual contours: steps of 1$\sigma$ starting at $\pm$2",\
            transform=axarr[0][0].transAxes)

filename= f"{plotfbase}_{nwalkers}{nburn}{nstep}_panels.png"
f.savefig(filename)
pl.close()
print("Done. Saved as", filename)

print("Generating phases plot...")
for j in range(mcmcresult['calphases_dset0'].shape[1]):
      pl.hist(mcmcresult['calphases_dset0'][:,j]*180/np.pi,bins=50,histtype='step')

pl.xlabel('Modelcal Phases, degrees')

filename=f"{plotfbase}_{nwalkers}{nburn}{nstep}_phases.png"
pl.savefig(filename)
pl.close()
print("Done. Saved as", filename)
