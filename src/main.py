import numpy as np
import matplotlib.pyplot as plt
import h5py
import pathlib

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
    for conf in stream.values():
        lattices = []
        for item in conf.items():
            # print(item[0])
            lattice = item[1]
            lattices.append(np.array(lattice).squeeze())

        # confs_list.append(np.mean(np.array(lattices), 0))
        confs_list.append(np.array(lattices)[0])
    confs = np.array(confs_list)

    # save np array
    with open(arrPath, 'wb') as arrFile:
        np.save(arrFile, confs)

    return confs


rawFilename = '../data/pion.local-local.u-gf-d-gi.px0_py0_pz0.h5'
arrFilename = '../data/confs.npy'
confs = load_mean_data(rawFilename, arrFilename, True)

# conf = confs[2][0::2]
conf = np.roll(confs[0][0::2], 10)
tau = np.arange(len(conf))
plt.plot(tau, conf)
plt.plot(tau, conf, 'x')
plt.savefig('../plot/test.png')