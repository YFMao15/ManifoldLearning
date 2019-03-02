import numpy as np
def generateNY(w,d):
    numOfSamples = w.shape[0]
    I = np.eye(numOfSamples,numOfSamples,dtype=np.float)
    M = np.matmul(np.transpose((I - w)),(I - w))
    punish = 1*I
    Eig, EigV = np.linalg.eig(M+20*punish)
    idx = Eig.argsort()
    EigV = EigV[:,idx]
    result = EigV[:,1:d+1]
    return result
