import numpy as np
from numpy import linalg as LA
from scipy.linalg import eigh
def mds(shortDisMap, d):
    N = shortDisMap.shape[0]
    #contruct tau(shortDisMap)
    S = np.multiply(shortDisMap,shortDisMap)
    H = np.eye(N,N)
    for idx in range(N):
        H[idx][idx] = 1 - 1.0/N

    tau = -np.matmul(np.matmul(H,S),H)/2.0

    #get first d eigenvalue and eigenvectors of tau
    Helper = 0
    w,v = LA.eigh(tau+Helper*np.eye(tau.shape[0], dtype=np.double))
    w = w - Helper
    w_idx = (-w).argsort()
    w = w[w_idx]
    v = v[:,w_idx]
    print(w)

    result = np.multiply(np.sqrt(w[0:d]),v[:,0:d])

    return result
