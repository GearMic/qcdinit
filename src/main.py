import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from qcdinit import *
from helpers import *


#def fit_fn(x, C, E):
#    T = 160
#    return C*(np.exp(-E * x) + np.exp(-(T - x) * E))

def fit_fn(x, C, E):
    T = 160
    return C * np.cosh(E * (T/2 - x))

def m_eff(data, tPos, tSpan):
    """calculate effective mass using time tPos and time range tSpan"""
    return 1/tSpan * np.arccosh( (data[tPos+tSpan] + data[tPos-tSpan]) / (2*data[tPos]) )

def cosh_fit(x, y, yCov, initialGuess, bounds=(-np.inf, np.inf), maxfev=600):
    # individual fit for parameter values
    popt, pcov, _, _, _ = optimize.curve_fit(fit_fn, x, y, initialGuess, yCov, full_output=True, absolute_sigma=True, bounds=bounds, maxfev=maxfev)

    # calculate chisq
    r = y - fit_fn(x, *popt)
    chisq = r.T @ sp.linalg.pinv(yCov) @ r # TODO: invert using singular value decomposition

    return popt, np.sqrt(np.diag(pcov)), chisq

def correlator_time_value_err(data, slicer=None):
    y = np.mean(data, 0)
    x = np.arange(len(data))
    yCov = np.cov(data.T, ddof=1) / confs.shape[0]

    if not (slicer is None):
        x = x[slicer]
        y = y[slicer]
        yCov = yCov[slicer, slicer]

    return x, y, yCov

def fn_correlator_fit(data, slicer, bounds=(-np.inf, np.inf)):
    x, y, yCov = correlator_time_value_err(data, slicer)

    popt, perr, chisq = cosh_fit(x, y, yCov, initialGuess, bounds)

    return np.array(tuple(popt) + tuple(perr) + (chisq,))

def fn_meff(data, tPos, tSpan):
    _, data, _ = correlator_time_value_err(data)

    return m_eff(data, tPos, tSpan)

def stability_plot(confs, plotMEff = True, tSpan=5):
    upperValues = np.array((40, 60, 80))
    maxUpper = np.max(upperValues)
    lowerLower = 10
    lowerDistance = 10
    upperColors = ('tab:purple', 'tab:orange', 'tab:red')

    fig, ax = plt.subplots()
    capsize = 3

    for j in range(len(upperValues)):
        upper = upperValues[j]
        upperColor = upperColors[j]

        lowerValues = np.arange(lowerLower, upper - lowerDistance)
        nFitIntervals = len(lowerValues)
        EArr, EErrArr = np.zeros(nFitIntervals), np.zeros(nFitIntervals)

        for i in range(nFitIntervals):
            slicer = slice(lowerValues[i], upper)

            # find E using correlator fit
            fitResult = fn_correlator_fit(confs, slicer, (0, np.inf))
            params = fitResult[:2] 
            paramsErr = fitResult[2:4] 

            EArr[i] = params[1]
            EErrArr[i] = paramsErr[1]

        # stability plot
        ax.errorbar(
            lowerValues, EArr, EErrArr,
            fmt='x', capsize=capsize, color=upperColor, zorder=6+j,
            label='upper bound = %i' % upper)
    
    if plotMEff:
        #TODO: calculate meff for all tPos values in one bootstrapping
        tPosValues = np.arange(lowerLower+tSpan+5, maxUpper-tSpan-10)
        ntPosValues = len(tPosValues)
        mEffArr, mEffErrArr = np.zeros(ntPosValues), np.zeros(ntPosValues)

        for i in range(ntPosValues):
            # find E using cosh formula ("effective mass")
            result, resultErr, _, _ = bootstrap_function(fn_meff, confs, 0, (tPosValues[i], tSpan), nSamples)

            mEffArr[i] = result
            mEffErrArr[i] = resultErr
    
        ax.errorbar(
            tPosValues, mEffArr, mEffErrArr,
            fmt='x', capsize=capsize, color='tab:blue', zorder=5,
            label=r'effective mass using $\tau=%i$' % tSpan
        )


    ax.legend()
    ax.set_title('Stability Plot')
    ax.set_xlabel('lower boundary of fit | t')
    ax.set_ylabel('resulting $E_o$')
    ax.grid()
    fig.savefig('plot/stability.pdf', **figArgs)


doBinErrorPlot = False
doStabilityPlot = True
plotMEff = True
initialGuess = (0.01, 0.01)
nStraps = 1000
nSamples = nStraps
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

# analyze
print('- analyze')

# stability depending on fit interval
if doStabilityPlot:
    print('- stability')
    stability_plot(confs, plotMEff)

# plot
print('-- plot')
#sliceA, sliceB = 16, 64
#sliceA, sliceB = 0, 48
sliceA, sliceB = 30, 50
sliceA, sliceB = 42, 80
slicer = slice(sliceA, sliceB)

fig, ax = plt.subplots()
ax.set_title('Correlator and fit in range %i, %i' % (slicer.start, slicer.stop))
ax.set_xlabel('$\\tau$')
ax.set_ylabel(r'Pion-Pion Correlator')
ax.grid()

ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())


p2p = np.mean(confs, 0) # final values for pion-pion 2pt-function
p2pCov = np.cov(confs.T, ddof=1) / confs.shape[0]
tau = np.arange(len(p2p))

## import comparison data
#compData = np.genfromtxt('data/CorrelationFunctionAngelo.csv', delimiter=',')
#p2pCov = np.genfromtxt('data/covarianzmatrixAngelo.csv', delimiter=',')
#tau = compData[:, 0]
#p2p = compData[:, 1]

ax.errorbar(tau, p2p, np.diag(p2pCov), fmt='.', color='xkcd:red', label='Data', zorder=2)


# fit to find pion mass
print('-- fit')
# extract fit range
dataLen = (sliceB-sliceA)

# fit using bootstrapping
result, resultErr, resultBootMean, resultArr = bootstrap_function(fn_correlator_fit, confs, 0, (slicer,), nSamples)
params = result[:2] 
paramsErr = result[2:4] 
chisq = result[-1]
C, E = params
Cerr, Eerr = paramsErr
paramsBootMean = resultBootMean[:2]
paramsBootErr = resultErr[:2]
paramsArr = resultArr[:, :2]

# print / export results
hbarc = 197.32698
mIdeal = 0.06821 * 139.57039 / hbarc
print('pion mass ideal: ', mIdeal)
resultsFrame = pd.DataFrame({
    '$X=$': ('C', 'E'),
    '$X$': params,
    r'$\delta X$': paramsErr,
    r'$\overline X_B$': paramsBootMean,
    r'$\delta X_{(B)}$': paramsBootErr,
    r'$\chi^2/\mathrm{dof}$': (chisq / (dataLen-2), '')
})
print("results:\n", resultsFrame.to_string())
resultsFrame.to_latex('latex/example_results.tex', index=False)
print('chisq/dof:', chisq/(dataLen-2))

# add fit line to plot
tauSlice = tau[slicer]
p2pSlice = p2p[slicer]
p2pCovSlice = p2pCov[slicer, slicer]

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

