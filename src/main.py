import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from qcdinit import *
from helpers import *


def fit_fn(x, C, E):
    T = 160
    return C*(np.exp(-E * x) + np.exp(-(T - x) * E))

#def fit_fn(x, C, E):
#    T = 160
#    return C * np.cosh(E * (T/2 - x))

#def fit_fn(x, *args):
#    return args[0] * np.cosh(-args[1] * (x-80))

def m_eff(data, tPos, tSpan):
    """calculate effective mass using time tPos and time range tSpan"""
    return 1/tSpan * np.arccosh( (data[tPos+tSpan] + data[tPos-tSpan]) / (2*data[tPos]) )

def stability_plot(tau, p2p, p2pCov):
    upper = 50
    lowerValues = np.arange(30, 40)
    nFitIntervals = len(lowerValues)
    EArr, EErrArr = np.zeros(nFitIntervals), np.zeros(nFitIntervals)
    
    for i in range(nFitIntervals):
        slice = (lowerValues[i], upper)
    
        tauSlice = tau[slice[0]:slice[1]]
        p2pSlice = p2p[slice[0]:slice[1]]
        p2pCovSlice = p2pCov[slice[0]:slice[1], slice[0]:slice[1]]
    
        params, paramsErr, _, _, _ = fit_bootstrap_correlated(
            fit_fn, tauSlice, p2pSlice, initialGuess, nStraps,
            yCov=p2pCovSlice, paramRange=((np.NINF, np.inf), (0, np.inf)), maxfev=maxfev
        )
        #params, paramsErr, _, _, _ = fit_bootstrap_correlated(
        #    fit_fn, tauSlice, p2pSlice, initialGuess, nStraps,
        #    yCov=p2pCovSlice, maxfev=maxfev
        #)

        EArr[i] = params[1]
        EErrArr[i] = paramsErr[1]
    
    # stability plot
    fig, ax = plt.subplots()
    ax.set_title('Stability Plot')
    ax.set_xlabel('lower boundary of fit / t')
    ax.set_ylabel('resulting $E_o$')
    ax.grid()
    ax.errorbar(lowerValues, EArr, EErrArr, fmt='x', color='tab:red', label='upper bound = %i' % upper)

    # add effective mass to plot
    tSpan = 5
    mEff = m_eff(p2p, lowerValues, tSpan)
    ax.plot(lowerValues, mEff, 'x', color = 'tab:blue', label=r'effective mass using $\tau=%i$' % tSpan)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 1]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower left')
    fig.savefig('plot/stability.pdf', **figArgs)


doBinErrorPlot = False
doStabilityPlot = False
initialGuess = (0.003, 0.04)
#initialGuess = (3.38e-3, 0.047)
#initialGuess = [1, -0.047]
initialGuess = (1, 0.01)
nStraps = 1000
figArgs = {'bbox_inches':'tight'}
maxfev = 600


# data importing
print('- load data')
rawFilename = 'data/pion.local-local.u-gf-d-gi.px0_py0_pz0.h5'
arrFilename = 'data/confs.npy'
confs = -load_mean_data(rawFilename, arrFilename, False).real

# export data for comparison
np.savetxt('data/confs_export.txt', confs.flatten())

halfIndex = int(np.floor(confs.shape[1]/2))
#confs = confs[:, :halfIndex]
# the two halves of a configuration are redundant so mean over them
confs = (confs + np.flip(np.roll(confs, -1, 1), 1) ) / 2
confs = confs[:, :halfIndex]


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
p2pCov = np.cov(confs.T, ddof=1) / confs.shape[0]
p2pErr = np.std(confs, 0) / np.sqrt(confs.shape[0])
tau = np.arange(len(p2p))

print(p2p)
print('-----')
print(p2pErr)
print('-----')
print(p2p.shape)
print(confs.shape)

# import comparison data
compData = np.genfromtxt('data/CorrelationFunctionAngelo.csv', delimiter=',')
p2pCov = np.genfromtxt('data/covarianzmatrixAngelo.csv', delimiter=',')
tau = compData[:, 0]
p2p = compData[:, 1]


# stability depending on fit interval
if doStabilityPlot:
    print('- stability')
    stability_plot(tau, p2p, p2pCov)


# plot
print('-- plot')
sliceA, sliceB = 16, 64
sliceA, sliceB = 30, 50
sliceA, sliceB = 0, 48
slicer = slice(sliceA, sliceB)

fig, ax = plt.subplots()
ax.set_title('Correlator and fit in range %i, %i' % (slicer.start, slicer.stop))
ax.set_xlabel('$\\tau$')
ax.set_ylabel(r'Pion-Pion Correlator')
ax.grid()

ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

ax.errorbar(tau, p2p, np.diag(p2pCov), fmt='.', color='xkcd:red', label='Data', zorder=2)


# fit to find pion mass
print('-- fit')
# extract fit range
dataLen = (sliceB-sliceA)
tauSlice = tau[slicer]
p2pSlice = p2p[slicer]
p2pCovSlice = p2pCov[slicer, slicer]

print('-----')
fullprint(p2pSlice)
print('-----')
fullprint(p2pCovSlice[:4, :])
print('-----')
print(p2pSlice.shape)

# fit using bootstrapping
params, paramsErr, chisq, paramsBootMean, paramsArr = fit_bootstrap_correlated(
    fit_fn, tauSlice, p2pSlice, initialGuess, nStraps, p2pCovSlice, None, maxfev=20000)
C, E = params
Cerr, Eerr = paramsErr

# print / export results
hbarc = 197.32698
mIdeal = 0.06821 * 139.57039 / hbarc
print('pion mass ideal: ', mIdeal)
resultsFrame = pd.DataFrame({
    '$X=$': ('C', 'E'),
    '$X$': params,
    r'$\delta X$': paramsErr,
    r'$\overline X_B$': paramsBootMean,
    r'$\chi^2/\mathrm{dof}$': (chisq / (dataLen-2), '')
})
print("results:\n", resultsFrame)
resultsFrame.to_latex('latex/example_results.tex', index=False)
print('chisq/dof:', chisq/(dataLen-2))

# add fit line to plot
xFit = tauSlice
yFit = fit_fn(xFit, C, E)
ax.plot(xFit, yFit, color='blue', label='Fit', zorder=5)

xFit2 = tauSlice
yFit2 = fit_fn(xFit, 0.003387, 0.04735)
ax.plot(xFit2, yFit2, color='blue', label='Fit', zorder=5)

# add bootstrap range to plot
print('-- bootstrap range plot')
rangeColor = 'xkcd:turquoise'

yLower, yUpper, yMean = np.zeros(dataLen), np.zeros(dataLen), np.zeros(dataLen)   
Cvalues, Evalues = paramsArr.T
for i in range(dataLen):
    t = xFit[i]

    yRange = fit_fn(t, Cvalues, Evalues)
    yLower[i], yUpper[i], yMean[i] = yRange.min(), yRange.max(), yRange.mean()

ax.fill_between(xFit, yLower, yUpper, alpha=0.5, label='bootstrap range', color=rangeColor, zorder=6)
ax.plot(xFit, yMean, label='bootstrap mean', color=rangeColor, zorder=4)

ax.legend()
fig.savefig('plot/correlator.pdf')

