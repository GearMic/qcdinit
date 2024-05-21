import numpy as np
import matplotlib.pyplot as plt
import h5py
import pathlib
import re

def load_mean_data(rawFilename, arrFilename, forceGenerate=False):
    # load saved np array and return if it exists
    arrPath = pathlib.Path(arrFilename)
    if not forceGenerate:
        if arrPath.is_file():
            with open(arrPath, 'rb') as arrFile:
                return np.load(arrFile)

    # mean over lattices of each configuration
    rawFile = h5py.File(rawFilename, 'r')
    stream = rawFile['stream_a']
    confs_list = []

    tRegex = re.compile('t\\d+_') # regular expression to find t value
    for conf in stream.values():
        lattices = []
        for item in conf.items():
            tMatch = tRegex.match(item[0])
            # extract t value
            if tMatch is None:
                print("ERROR: couldn't find t value in %s" % item[0])
            t = int(tMatch.group()[1:-1])

            #convert to complex array
            latticeFloat = np.array(item[1]).squeeze()
            latticeComplex = np.empty(len(latticeFloat)//2, np.cdouble)
            latticeComplex.real = latticeFloat[0::2]
            latticeComplex.imag = latticeFloat[1::2]
            # print(latticeComplex)
            # print(item[0], t, np.array(lattice).shape)
            # add rolled array to the list
            lattices.append(np.roll(latticeComplex, -t))

        # confs_list.append(np.mean(np.array(lattices), 0))
        confs_list.append(np.array(lattices)[5])
    confs = np.array(confs_list)

    # save np array
    with open(arrPath, 'wb') as arrFile:
        np.save(arrFile, confs)

    return confs


rawFilename = '../data/pion.local-local.u-gf-d-gi.px0_py0_pz0.h5'
arrFilename = '../data/confs.npy'
confs = load_mean_data(rawFilename, arrFilename, True)

# conf = confs[0][0::2]
conf = confs[0]
tau = np.arange(len(conf))
plt.plot(tau, conf)
plt.plot(tau, conf, 'x')
plt.savefig('../plot/test.png')