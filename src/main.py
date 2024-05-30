import numpy as np
import matplotlib.pyplot as plt

from qcdinit import *
from montecarlo import *

def p2p_fit_bootstrapping(tau, p2p, fig, ax):
    # plot
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())

    # ax.errorbar(tau, p2p, p2pErr, fmt='.')
    ax.errorbar(tau, p2p, p2pErr)

    # fit to find pion mass
    def fit_fn(x, C, E):
        T = 160 # TODO: Why do we need this?
        return C*(np.exp(-E * x) + np.exp(-(T - x) * E))

    params, paramsErr, _ = fit_bootstrap(
        fit_fn, tau, p2p, (1.5, 0.1), 10000, p2pErr, None)
    C, E = params
    CErr, EErr = paramsErr
    prerr(C, CErr, 'C')
    prerr(E, EErr, 'E (pion mass)')

    hbarc = 197.32698
    mIdeal = 0.06821 * 139.57039 / hbarc
    print('pion mass ideal: ', mIdeal)

    # add fit line to plot
    xFit = tau
    yFit = fit_fn(xFit, C, E)

    ax.plot(xFit, yFit, color='xkcd:crimson')

    return C, E, CErr, EErr



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
ax.set_title('Correlator fit')
ax.set_xlabel('$\tau$ [lattice spacing]')
ax.set_ylabel('$E_o$')
ax.grid()

ax.set_yscale('log')
ax.yaxis.set_major_formatter(plt.ScalarFormatter())

ax.errorbar(tau, p2p, p2pErr, fmt='x', label='Data')

# fit to find pion mass
initialGuess = (1.5, 0.1)
nStraps = 2000

def fit_fn(x, C, E):
    T = 160 # TODO: Why do we need this?
    return C*(np.exp(-E * x) + np.exp(-(T - x) * E))

slice = (8, 58)
tauSlice = tau[slice[0]:slice[1]]
p2pSlice = p2p[slice[0]:slice[1]]
p2pErrSlice = p2pErr[slice[0]:slice[1]]

params, paramsErr, _ = fit_bootstrap(
    fit_fn, tauSlice, p2pSlice, initialGuess, nStraps, p2pErrSlice, None)
C, E = params
CErr, EErr = paramsErr
prerr(C, CErr, 'C')
prerr(E, EErr, 'E (pion mass)')

hbarc = 197.32698
mIdeal = 0.06821 * 139.57039 / hbarc
print('pion mass ideal: ', mIdeal)

# add fit line to plot
xFit = tauSlice
yFit = fit_fn(xFit, C, E)

ax.plot(xFit, yFit, color='xkcd:crimson', label='Fit')

ax.legend()
fig.savefig('../plot/visual_log_2.pdf')


# stability depending on fit interval
upper = 58
lowerValues = np.arange(1, 15)
nFitIntervals = len(lowerValues)
EArr, EErrArr = np.zeros(nFitIntervals), np.zeros(nFitIntervals)

for i in range(nFitIntervals):
    slice = (lowerValues[i], upper)

    tauSlice = tau[slice[0]:slice[1]]
    p2pSlice = p2p[slice[0]:slice[1]]
    p2pErrSlice = p2pErr[slice[0]:slice[1]]

    params, paramsErr, _ = fit_bootstrap(
        fit_fn, tauSlice, p2pSlice, initialGuess, nStraps, p2pErrSlice, None)
    EArr[i] = params[1]
    EErr = paramsErr[1]

# stability plot
fig, ax = plt.subplots()
ax.set_title('stability plot')
ax.set_xlabel('lower boundary of fit [lattice spacing]')
ax.set_ylabel('resulting $E_o$')
ax.grid()
ax.errorbar(lowerValues, EArr, EErrArr, fmt='x', color='tab:red')
fig.savefig('../plot/stability.pdf')


# TODO: don't halve fit interval