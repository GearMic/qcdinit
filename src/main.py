import numpy as np
import matplotlib.pyplot as plt

from qcdinit import *
from montecarlo import *

# data importing
rawFilename = '../data/pion.local-local.u-gf-d-gi.px0_py0_pz0.h5'
arrFilename = '../data/confs.npy'
confs = load_mean_data(rawFilename, arrFilename, False)
# the second half of each configuration is redundant
confs = np.abs(confs[:, :confs.shape[1]//2]) 

# bin error plot
fig, ax = plt.subplots()
for i in np.arange(0, confs.shape[1], 20):
    obs = confs[:, i]
    bin_error_plot(obs, fig, ax, "$t=%i$" % i, 10)
fig.savefig('../plot/bin_error.pdf')

# bin and mean
binsize = 1 # visually determined from error plot
confs = bin_mean(confs, binsize)
p2p = np.mean(confs, 0) # final values for pion-pion 2pt-function
p2pErr = expectation_error_estimate(confs, 0)

# plot
tau = np.arange(len(p2p))

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.yaxis.set_major_formatter(plt.ScalarFormatter())

# ax.errorbar(tau, p2p, p2pErr, fmt='.')
ax.errorbar(tau, p2p, p2pErr)

# fit to find pion mass
def fit_fn(x, C, E):
    T = 160 # TODO: Why do we need this?
    return C*(np.exp(-E * x) + np.exp(-(T - x) * E))

slice = (8, 58)
tau = tau[slice[0]:slice[1]]
p2p = p2p[slice[0]:slice[1]]
p2pErr = p2pErr[slice[0]:slice[1]]

params, paramsErr, _ = fit_bootstrap(
    fit_fn, tau, p2p, (1.5, 0.1), 10000, p2pErr, None)
C, E = params
CErr, EErr = paramsErr
prerr(C, CErr, 'C')
prerr(E, EErr, 'E (pion mass)')

hbarc = 1.054571 * 299702458 / 1.602176 * (1e-34*1e19*1e15*1e6)
mIdeal = 0.06821 * 139.57039 / hbarc
print('pion mass ideal: ', mIdeal)

# add fit line to plot
xFit = tau
yFit = fit_fn(xFit, C, E)

ax.plot(xFit, yFit, color='xkcd:crimson')
fig.savefig('../plot/visual_log.pdf')
