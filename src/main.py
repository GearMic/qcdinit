import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from qcdinit import *
from montecarlo import *


def fit_fn(x, C, E):
    T = 160
    return C*(np.exp(-E * x) + np.exp(-(T - x) * E))

doBinErrorPlot = False
initialGuess = (1.5, 0.1)
nStraps = 1000


# data importing
print('- load data')
rawFilename = 'data/pion.local-local.u-gf-d-gi.px0_py0_pz0.h5'
arrFilename = 'data/confs.npy'
confs = load_mean_data(rawFilename, arrFilename, False).real
halfIndex = int(np.floor(confs.shape[1]/2))
confs = confs[:, :halfIndex]
## the two halves of a configuration are redundant so mean over them
#confs = (confs[:, :halfIndex] + confs[:, -halfIndex:]) / 2 # TODO: not working properly


# bin error plot
if doBinErrorPlot:
    print('- bin error plot')
    fig, ax = plt.subplots()
    for i in np.arange(0, confs.shape[1], 20):
        obs = confs[:, i]
        bin_error_plot(obs, fig, ax, "$t=%i$" % i, 10)
    fig.savefig('plot/bin_error.pdf')


# prepare data
print('- prepare data')
binsize = 1 # visually determined from error plot
confs = bin_mean(confs, binsize)
p2p = np.mean(confs, 0) # final values for pion-pion 2pt-function
p2pErr = expectation_error_estimate(confs, 0)
tau = np.arange(len(p2p))


# stability depending on fit interval
print('- stability')
upper = 50
lowerValues = np.arange(8, 40)
nFitIntervals = len(lowerValues)
EArr, EErrArr = np.zeros(nFitIntervals), np.zeros(nFitIntervals)

for i in range(nFitIntervals):
    slice = (lowerValues[i], upper)

    tauSlice = tau[slice[0]:slice[1]]
    p2pSlice = p2p[slice[0]:slice[1]]
    p2pErrSlice = p2pErr[slice[0]:slice[1]]

    params, paramsErr, _, paramsBootMean, _ = fit_bootstrap(
        fit_fn, tauSlice, p2pSlice, initialGuess, nStraps,
        yErr=p2pErrSlice, paramRange=((np.NINF, np.inf), (0, np.inf))
    )
    EArr[i] = params[1]
    EErrArr[i] = paramsErr[1]


# stability plot
fig, ax = plt.subplots()
ax.set_title('stability plot')
ax.set_xlabel('lower boundary of fit')
ax.set_ylabel('resulting $E_o$')
ax.grid()
ax.errorbar(lowerValues, EArr, EErrArr, fmt='x', color='tab:red')
#ax.errorbar(range(nFitIntervals), EArr, EErrArr, fmt='x', color='tab:red')
fig.savefig('plot/stability.pdf')
print(EArr)
print(EErrArr)


# plot
print('-- plot')

fig, ax = plt.subplots()
ax.set_title('Correlator fit')
ax.set_xlabel('$\\tau$ [lattice spacing]')
ax.set_ylabel(r'Pion-Pion Correlator $\cdot (-1)$')
ax.grid()

ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

ax.errorbar(tau, -p2p, p2pErr, fmt='x', label='Data')


# fit to find pion mass
print('-- fit')
slice = (30, 50) # boundaries determined visually (from stability plot)
tauSlice = tau[slice[0]:slice[1]]
p2pSlice = p2p[slice[0]:slice[1]]
p2pErrSlice = p2pErr[slice[0]:slice[1]]

params, paramsErr, chisq, paramsBootMean, paramsArr = fit_bootstrap(
    fit_fn, tauSlice, p2pSlice, initialGuess, nStraps, p2pErrSlice, None)
C, E = params
Cerr, Eerr = paramsErr
#prerr(C, Cerr, 'C')
#prerr(E, Eerr, 'E (pion mass)')

# print / export results
hbarc = 197.32698
mIdeal = 0.06821 * 139.57039 / hbarc
print('pion mass ideal: ', mIdeal)
resultsFrame = pd.DataFrame({
    '$X=$': ('C', 'E'),
    '$X$': params,
    r'$\delta X$': paramsErr,
    r'$\overline X_B$': paramsBootMean,
    r'$\Chi^2$': (chisq, '')
})
print("results:\n", resultsFrame)
resultsFrame.to_latex('latex/example_results.tex')

# add fit line to plot
xFit = tauSlice
yFit = fit_fn(xFit, C, E)

ax.plot(xFit, -yFit, color='xkcd:crimson', label='Fit')

ax.legend()
fig.savefig('plot/visual_log_2.pdf')

# add bootstrap range to plot
print('-- bootstrap range plot')
## TODO
