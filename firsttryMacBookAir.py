# scikit-learn first try
homeDir = '/Users/Simon/Dropbox/NeuroTon/Matrizen4Simon/' # location of the connectivity matrices
nROIs = 4 # number of ROIs in the connectivity matrix
import numpy as np
import os
from sklearn import svm
connMatList = os.listdir(homeDir)
connMatList = sorted(connMatList)
samples = np.full([len(connMatList), int((nROIs*(nROIs-1))/2)], -99.0)
for i in range(0,len(connMatList)):
    temp = np.loadtxt(os.path.join(homeDir, connMatList[i]))
    flatTemp = temp[np.triu_indices(n = temp.shape[1], k = 1)]
    samples[i] = flatTemp
    
print(samples)
clf = svm.SVC(gamma=0.001, C=100.)
