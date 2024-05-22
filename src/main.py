import numpy as np
import matplotlib.pyplot as plt

import qcdinit

rawFilename = '../data/pion.local-local.u-gf-d-gi.px0_py0_pz0.h5'
arrFilename = '../data/confs.npy'
confs = qcdinit.load_mean_data(rawFilename, arrFilename, False)

# conf = confs[0][0::2]
conf = confs[0]
tau = np.arange(len(conf))
plt.plot(tau, np.abs(conf))
plt.plot(tau, np.abs(conf), 'x')
plt.savefig('../plot/test.png')