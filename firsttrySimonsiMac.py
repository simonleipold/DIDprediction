# scikit-learn first try
homeDir = '/Users/simonleipold/Dropbox/NeuroTon/Matrizen4Simon' # location of the connectivity matrices
nROIs = 4 # number of ROIs in the connectivity matrix
# import packages needed for classification
import numpy as np
import os
from sklearn import svm

rawFiles = os.listdir(homeDir)
wantedFiles = []
for f in rawFiles:
    if 'tempConn' in f:
        wantedFiles.append(f) # exclude unwanted files from filelist

connList = sorted(wantedFiles)

targets = []
for f in connList:
    if 'AP' in f:
        targets.append('AP')
    elif 'RP' in f:
        targets.append('RP')

samples = np.full([len(connList), int((nROIs*(nROIs-1))/2)], -99.0)
for i in range(0,len(connList)):
    temp = np.loadtxt(os.path.join(homeDir, connList[i]))
    flatTemp = temp[np.triu_indices(n = temp.shape[1], k = 1)]
    samples[i] = flatTemp
#print(samples)
clf = svm.SVC(gamma=0.001, C=1.)
