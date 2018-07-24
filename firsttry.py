# scikit-learn first try
homeDir = '/Users/simonleipold/Dropbox/NeuroTon/Matrizen4Simon/'
import numpy as np
import os
connMatList = os.listdir(homeDir)
for subj in connMatList:
    if 'tempConn' in subj:
        print(subj)
#dat = np.loadtxt('/Users/simonleipold/Dropbox/NeuroTon/Matrizen4Simon/tempConn_1.txt')
#print(dat)
#flatDat = dat[np.triu_indices(n = dat.shape[1], k = 1)]
#print(flatDat)
