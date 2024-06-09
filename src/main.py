import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from qcdinit import *
from montecarlo import *


doBinErrorPlot = False

# data importing
print('- load data')
rawFilename = 'data/pion.local-local.u-gf-d-gi.px0_py0_pz0.h5'
arrFilename = 'data/confs.npy'
confs = load_mean_data(rawFilename, arrFilename, False)
# the second half of each configuration is redundant TODO: is this correct?
confs = confs[:, :confs.shape[1]//2].real

# bin error plot
if doBinErrorPlot:
    print('- bin error plot')
    fig, ax = plt.subplots()
    for i in np.arange(0, confs.shape[1], 20):
        obs = confs[:, i]
        bin_error_plot(obs, fig, ax, "$t=%i$" % i, 10)
    fig.savefig('plot/bin_error.pdf')

# bin and mean
print('- mean bins')
binsize = 1 # visually determined from error plot
confs = bin_mean(confs, binsize)
p2p = np.mean(confs, 0) # final values for pion-pion 2pt-function
p2pErr = expectation_error_estimate(confs, 0)


# plot
print('- plot and fit')
tau = np.arange(len(p2p))

fig, ax = plt.subplots()
ax.set_title('Correlator fit')
ax.set_xlabel('$\\tau$ [lattice spacing]')
ax.set_ylabel(r'Pion-Pion Correlator $\cdot (-1)$')
ax.grid()

ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

ax.errorbar(tau, -p2p, p2pErr, fmt='x', label='Data')

# fit to find pion mass
initialGuess = (1.5, 0.1)
nStraps = 1000

def fit_fn(x, C, E):
    T = 160
    return C*(np.exp(-E * x) + np.exp(-(T - x) * E))

slice = (8, 58)
tauSlice = tau[slice[0]:slice[1]]
p2pSlice = p2p[slice[0]:slice[1]]
p2pErrSlice = p2pErr[slice[0]:slice[1]]

params, paramsErr, _, _ = fit_bootstrap(
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

ax.plot(xFit, -yFit, color='xkcd:crimson', label='Fit')

ax.legend()
fig.savefig('plot/visual_log_2.pdf')


# stability depending on fit interval
print('- stability')
upper = 58
lowerValues = np.arange(1, 15)
nFitIntervals = len(lowerValues)
lowerValues = np.full(nFitIntervals, 15) # debugging purposes
EArr, EErrArr = np.zeros(nFitIntervals), np.zeros(nFitIntervals)

for i in range(nFitIntervals):
    slice = (lowerValues[i], upper)

    tauSlice = tau[slice[0]:slice[1]]
    p2pSlice = p2p[slice[0]:slice[1]]
    p2pErrSlice = p2pErr[slice[0]:slice[1]]

#    C, E, Cerr, Eerr = cosh_fit_bootstrap(
#        tauSlice, p2pSlice, initialGuess, nStraps, p2pErrSlice, None)
#    params = (C, E)
#    paramsEerr = (Cerr, Eerr)

    params, paramsErr, _, paramsBootMean = fit_bootstrap(
        fit_fn, tauSlice, p2pSlice, initialGuess, nStraps, yErr=p2pErrSlice, paramRange=((np.NINF, np.inf), (0, np.inf)))
    EArr[i] = params[1]

##    EArr[i] = paramsBootMean[1] ##
    EErrArr[i] = paramsErr[1]

# print(EErr)
# print(EErrArr)

# stability plot
fig, ax = plt.subplots()
ax.set_title('stability plot')
ax.set_xlabel('lower boundary of fit [lattice spacing]')
ax.set_ylabel('resulting $E_o$')
ax.grid()
# ax.errorbar(lowerValues, EArr, EErrArr, fmt='x', color='tab:red')
ax.errorbar(range(nFitIntervals), EArr, EErrArr, fmt='x', color='tab:red')
fig.savefig('plot/stability.pdf')
print(EArr)
print(EErrArr)

# TODO: inspect "quantization of errors"
