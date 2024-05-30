import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize


def bin_normalized(data, n_bins, xlower, xupper):
    rows, cols = data.shape
    bin_size = (xupper - xlower) / n_bins

    bins, edges = np.histogram(data, n_bins, (xlower, xupper))
    bins_x = edges[:-1] + bin_size / 2
    bins_y = bins / bin_size / (rows*cols)

    return bins_x, bins_y

def check_nd(array: np.ndarray, n: int):
    """print Error if array is not n-dimensional"""
    m = array.ndim
    if m != n:
        print("ERROR: array should be %id, but is %id." % (n, m))

def autocorrelation_estimator(obs: np.ndarray, t: int, obs_mean: np.ndarray = None, periodic: bool = False):
    """
    Calculates mean correlation of lattice point with the lattice point at t later (later in lattice time) 
    if obs is 2d, does so individually for each row of obs (so along axis 1).
    'obs' stands for observable.
    TODO: rewrite in terms of eq(31) from Monte Carlo Errors Paper
    TODO: proper implementation for n-dim
    TODO: properly understand this, check if error is taken along the correct axis
    """

    if obs.ndim == 1:
        obs = np.expand_dims(obs, 0)
    
    if obs_mean is None:
        obs_mean = np.mean(obs, 1)

    mean_tiled = np.tile(obs_mean, (obs.shape[1], 1)).T
    deviation = obs - mean_tiled # deviation from mean for each row
    obs_correlations = ((deviation) * np.roll((deviation), -t, 1)) # multiply each value at i with the value at i+t

    # cut away the values past the end of the array if periodic is false
    if not periodic and t != 0:
        obs_correlations = obs_correlations[:, :-t]

    return np.squeeze(np.mean(obs_correlations, 1))

def autocorrelation_range_mean(obs: np.ndarray, N: int, obs_mean: np.ndarray = None, periodic: bool = False):
    """
    DEPRECATED
    calculate autocorrelation for t-values up to N.
    if obs is 2d, then the mean of the correlations is taken for each t.
    """

    trange = range(N)

    corr = np.zeros(N)
    for t in trange:
        corr[t] = np.mean(autocorrelation_estimator(obs, t, obs_mean, periodic))

    return np.array(trange), corr

def autocorrelation_range(obs: np.ndarray, N: int, obs_mean: np.ndarray = None, periodic: bool = False):
    """
    calculate autocorrelation for t-values up to N (done along axis 1 if obs if 2d).
    if obs is 2d, then the mean of the correlations is taken for each t.
    """

    if obs.ndim == 1:
        rows = 1
    else:
        rows = obs.shape[0]

    trange = range(N)

    corr = np.zeros((rows, N))
    for t in trange:
        corr[:, t] = autocorrelation_estimator(obs, t, obs_mean, periodic)

    return np.array(trange), np.squeeze(corr)

def ensemble_autocorrelation_mean(ensemble: np.ndarray, a: float):
    """
    DEPRECATED
    autocorrelation over an ensemble with lattice distance a and periodic boundary conditions.
    returns array of distance values and array of corresponding correlations.
    """

    N = int(np.ceil(ensemble.shape[1] / 2))
    trange, corr = autocorrelation_range_mean(ensemble, N, periodic=True)

    return a*np.array(trange), corr

def ensemble_autocorrelation(ensemble: np.ndarray):
    """
    autocorrelation over an ensemble with lattice distance a and periodic boundary conditions.
    returns array of distance values and array of corresponding correlations.
    """

    mean = np.zeros(ensemble.shape[0]) # TODO: implement this properly; why is the mean not zero anyways?
    N = int(np.ceil(ensemble.shape[1] / 2))
    trange, corr = autocorrelation_range(ensemble, N, mean, periodic=True)

    return trange, corr

def bin_mean(obs: np.ndarray, size: int, axis: int = 0):
    """
    turn size values of obs into one value by calculating the mean.
    """
    if size==1: return obs

    obs = obs.swapaxes(0, axis)
    n = len(obs)
    n_bins = np.floor_divide(n, size) 

    binned_shape = list(obs.shape)
    binned_shape[0] = n_bins
    obs_binned = np.zeros(binned_shape)
    for i in range(n_bins):
        bin_start = i*size
        obs_binned[i] = np.mean(obs[bin_start:bin_start+size], 0)

    obs_binned = obs_binned.swapaxes(0, axis)
    return obs_binned

def expectation_error_estimate(obs: np.ndarray, axis: int):
    """
    implements (34) from Statistik I along axis.
    TODO: test for axis!=0
    """
    obs = obs.swapaxes(0, axis)
    N = len(obs)

    mean = np.mean(obs, axis)
    if obs.ndim == 1:
        mean_spread = mean
    else:
        mean_spread = np.full(obs.shape, mean)

    var = 1 / (N-1) * np.sum((obs-mean)**2, 0)

    error = np.sqrt(var) / np.sqrt(N)
    return error

def errors_of_binned(obs: np.ndarray, max_size: int):
    """
    calculates errors of obs for increasing bin size.
    Error calculated as error of mean.
    """

    # calculate standard deviation for different bin sizes
    binsize, error = [i+1 for i in range(max_size)], []
    for size in binsize:
        obs_binned = bin_mean(obs, size)
        error.append(expectation_error_estimate(obs_binned, 0))

    return np.array(binsize), np.array(error)

def bin_error_plot(data, fig, ax, label=None, max_binsize=100, filename=None, dpi=300):
    """
    Plot of error against bin size using fig and ax.
    If filename=None then the plot has to be saved manually.
    """
    binsize, error = errors_of_binned(data, max_binsize)

    ax.plot(binsize, error, '.', label=label)
    ax.set_xlim(left=1, right=max_binsize)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Bin Size")
    ax.set_ylabel("Naive Error")
    ax.set_title("Error against bin size")
    ax.grid()
    ax.legend()

    if not (filename is None):
        fig.savefig(filename, dpi=dpi)


def exp_fit_bootstrap(x, y, initialGuess, nStraps, yErr=None, sliceLen=None):
    """
    exponential fit of the form f(x) = a * exp(-b * x).
    returns a, b, a_err, b_err. Errors calculated using bootstrapping.
    if yErr is None, yErr is initialized to an array of ones.
    """

    def exp_fn(x: float, a: float, b: float):
        return a * np.exp(-b*x)

    # prepare data arrays
    dataLen = len(x)
    if yErr is None:
        yErr = np.ones(dataLen)
    if sliceLen is None:
        sliceLen = len(x)
    x = x[:sliceLen]
    y = y[:sliceLen]
    yErr = yErr[:sliceLen]
    iRange = range(dataLen)

    # individual fit for parameter values
    popt, pcov, infodict, _, _ = optimize.curve_fit(exp_fn, x, y, initialGuess, yErr, full_output=True)
    a, b = popt
    chisq = np.sum(infodict['fvec']**2)

    # bootstrapping for parameter errors
    aArr, bArr = np.zeros(nStraps), np.zeros(nStraps)
    for i in range(nStraps):
        sampleIndex = np.random.choice(iRange, dataLen)
        xSample = x[sampleIndex]
        ySample = y[sampleIndex]
        yErrSample = yErr[sampleIndex]

        poptBoot, _ = optimize.curve_fit(
            exp_fn, xSample, ySample, initialGuess, yErrSample)
        aArr[i], bArr[i] = poptBoot

    aErr = np.std(aArr, ddof=1)
    bErr = np.std(bArr, ddof=1)

    return a, b, aErr, bErr, chisq


def fit_bootstrap(fit_fn, x, y, initialGuess, nStraps, yErr=None, sliceLen=None):
    """
    exponential fit of the form f(x) = a * exp(-b * x).
    returns a, b, a_err, b_err. Errors calculated using bootstrapping.
    if yErr is None, yErr is initialized to an array of ones.
    """

    # prepare data arrays
    dataLen = len(x)
    if yErr is None:
        yErr = np.ones(dataLen)
    if sliceLen is None:
        sliceLen = len(x)
    x = x[:sliceLen]
    y = y[:sliceLen]
    yErr = yErr[:sliceLen]
    iRange = range(dataLen)

    # individual fit for parameter values
    params, _, infodict, _, _ = optimize.curve_fit(fit_fn, x, y, initialGuess, yErr, full_output=True)
    chisq = np.sum(infodict['fvec']**2)

    # bootstrapping for parameter errors
    paramsArr = np.empty((nStraps, len(params)))
    for i in range(nStraps):
        sampleIndex = np.random.choice(iRange, dataLen)
        xSample = x[sampleIndex]
        ySample = y[sampleIndex]
        yErrSample = yErr[sampleIndex]

        paramsBoot, _ = optimize.curve_fit(
            fit_fn, xSample, ySample, initialGuess, yErrSample)
        paramsArr[i] = np.array(paramsBoot)

    paramsErr = tuple(np.std(paramsArr, 0, ddof=1))

    return params, paramsErr, chisq