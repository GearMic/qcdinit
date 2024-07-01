import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import scipy as sp

from qcdinit import *
from montecarlo import *


def fit_fn(x, C, E):
    T = 160
    return C*(np.exp(-E * x) + np.exp(-(T - x) * E))

def m_eff(data, tPos, tSpan):
    """calculate effective mass using time tPos and time range tSpan"""
    return 1/tSpan * np.arccosh( (data[tPos+tSpan] + data[tPos-tSpan]) / (2*data[tPos]) )

def stability_plot(tau, p2p, p2pErr):
    upper = 50
    lowerValues = np.arange(30, 40)
    nFitIntervals = len(lowerValues)
    EArr, EErrArr = np.zeros(nFitIntervals), np.zeros(nFitIntervals)
    
    for i in range(nFitIntervals):
        slice = (lowerValues[i], upper)
    
        tauSlice = tau[slice[0]:slice[1]]
        p2pSlice = p2p[slice[0]:slice[1]]
        p2pErrSlice = p2pErr[slice[0]:slice[1]]
        p2pCovSlice = p2pCov[slice[0]:slice[1], slice[0]:slice[1]]
    
        # params, paramsErr, _, _, _ = fit_bootstrap(
        #     fit_fn, tauSlice, p2pSlice, initialGuess, nStraps,
        #     yErr=p2pErrSlice, paramRange=((np.NINF, np.inf), (0, np.inf))
        # )

        # params, paramsErr, _, _, _ = fit_bootstrap(
        #     fit_fn, tauSlice, p2pSlice, initialGuess, nStraps,
        #     yErr=np.diag(p2pCovSlice), paramRange=((np.NINF, np.inf), (0, np.inf))
        # )

        params, paramsErr, _, _, _ = fit_bootstrap_correlated(
            fit_fn, tauSlice, p2pSlice, initialGuess, nStraps,
            yCov=p2pCovSlice, paramRange=((np.NINF, np.inf), (0, np.inf))
        )

        EArr[i] = params[1]
        EErrArr[i] = paramsErr[1]
    
    # stability plot
    fig, ax = plt.subplots()
    ax.set_title('Stability Plot')
    ax.set_xlabel('lower boundary of fit')
    ax.set_ylabel('resulting $E_o$')
    ax.grid()
    ax.errorbar(lowerValues, EArr, EErrArr, fmt='x', color='tab:red', label='upper bound = %i' % upper)

    # add effective mass to plot
    tSpan = 5
    mEff = m_eff(p2p, lowerValues, tSpan)
    print(mEff)
    #lowerValues = lowerValues[0:len(mEff)]
    print(mEff.shape)
    ax.plot(lowerValues, mEff, 'x', color = 'tab:blue', label=r'effective mass using $t=\mathrm{lower bound},\enspace\tau=%i$' % tSpan)
    
    ax.legend()
    fig.savefig('plot/stability.pdf', **figArgs)


doBinErrorPlot = False
doStabilityPlot = False
initialGuess = (1.5, 0.1)
nStraps = 1000
figArgs = {'bbox_inches':'tight'}


# data importing
print('- load data')
rawFilename = 'data/pion.local-local.u-gf-d-gi.px0_py0_pz0.h5'
arrFilename = 'data/confs.npy'
confs = load_mean_data(rawFilename, arrFilename, False).real

# export data for comparison
np.savetxt('data/confs_export.txt', confs.flatten())

halfIndex = int(np.floor(confs.shape[1]/2))
confs = confs[:, :halfIndex]
## TODO TODO: the two halves of a configuration are redundant so mean over them
#confs = (confs[:, :halfIndex] + confs[:, -halfIndex:]) / 2 # 


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
p2pErr = np.std(confs, 0, ddof=1)
p2pCov = np.cov(confs.T, ddof=1)
print(p2pCov.shape)
p2pErr = expectation_error_estimate(confs, 0)
tau = np.arange(len(p2p))

# stability depending on fit interval
if doStabilityPlot:
    print('- stability')
    stability_plot(tau, p2p, p2pErr)


# plot
print('-- plot')
slice = (30, 50) # boundaries determined visually (from stability plot)

fig, ax = plt.subplots()
ax.set_title('Correlator and fit in range %s' % str(slice))
ax.set_xlabel('$\\tau$')
ax.set_ylabel(r'Pion-Pion Correlator $\cdot (-1)$')
ax.grid()

ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

ax.errorbar(tau, -p2p, p2pErr, fmt='.', color='xkcd:red', label='Data', zorder=2)


# fit to find pion mass
print('-- fit')
dataLen = slice[1]-slice[0]
tauSlice = tau[slice[0]:slice[1]]
p2pSlice = p2p[slice[0]:slice[1]]
p2pErrSlice = p2pErr[slice[0]:slice[1]]
p2pCovSlice = p2pCov[slice[0]:slice[1], slice[0]:slice[1]]

#params, paramsErr, chisq, paramsBootMean, paramsArr = fit_bootstrap(
#    fit_fn, tauSlice, p2pSlice, initialGuess, nStraps, p2pErrSlice, None)
params, paramsErr, chisq, paramsBootMean, paramsArr = fit_bootstrap_correlated(
    fit_fn, tauSlice, p2pSlice, initialGuess, nStraps, p2pCovSlice, None)
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
    r'$\chi^2/\mathrm{dof}$': (chisq / (dataLen-2), '')
})
print("results:\n", resultsFrame)
resultsFrame.to_latex('latex/example_results.tex')

# add fit line to plot
xFit = tauSlice
yFit = fit_fn(xFit, C, E)

ax.plot(xFit, -yFit, color='blue', label='Fit', zorder=5)

# add bootstrap range to plot
print('-- bootstrap range plot')
rangeColor = 'xkcd:turquoise'

yLower, yUpper, yMean = np.zeros(dataLen), np.zeros(dataLen), np.zeros(dataLen)   
Cvalues, Evalues = paramsArr.T
Cvalues = -Cvalues # make values positive for plot
for i in range(dataLen):
    t = xFit[i]

    yRange = fit_fn(t, Cvalues, Evalues)
    yLower[i], yUpper[i], yMean[i] = yRange.min(), yRange.max(), yRange.mean()

ax.fill_between(xFit, yLower, yUpper, alpha=0.5, label='bootstrap range', color=rangeColor, zorder=6)
ax.plot(xFit, yMean, label='bootstrap mean', color=rangeColor, zorder=4)

ax.legend()
fig.savefig('plot/correlator.pdf')

