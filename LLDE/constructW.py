import numpy as np
from readYale import readYale
from generateMap import generateMap

def constructW(data0,K,distanceType):
    data = np.double(data0)
    map = generateMap(data[:,0:data.shape[1]], distanceType)
    numOfSamples = map.shape[0]
    numOfDims = data.shape[1]
    w = np.zeros(map.shape, dtype=np.float)
    #construct w
    for i in range(numOfSamples):
        print(i,'/',numOfSamples)
        #construct w[i,:]
        #find k neighbors
        idx = map[i,:].argsort()
        w[i,idx[K+1:numOfSamples]] = 0
        penalize = np.eye(K, K) * 0.001
        C = np.zeros((K, K))

        # construct C
        for j in range(K):
            for k in range(K):
                C[j][k] = np.matmul(data[i,0:numOfDims] - data[idx[j + 1], 0:numOfDims], data[i,0:numOfDims] - data[idx[k + 1], 0:numOfDims])

        invC = np.linalg.inv(C+penalize)
        for j in range(K):
            alpha = 0
            beta = 0
            for k in range(K):
                alpha = alpha + invC[j][k]

            for l in range(K):
                for m in range(K):
                    beta = beta+invC[l][m]

            w[i][idx[j+1]] = alpha*1.0/beta

    return w
