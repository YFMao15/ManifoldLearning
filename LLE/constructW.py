import numpy as np
from readYale import readYale
from generateMap import generateMap

def constructW(map,data,K):
    numOfSamples = map.shape[0]
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

        '''
        #construct C
        for j in range(K):
            for k in range(K):
                C[j][k] = np.matmul(data[idx[j+1],0:numOfDims],data[idx[k+1],0:numOfDims])

        #get alpha and beta
        alpha = 1
        beta = 0
        invC = np.linalg.inv(C+penalize)
        for j in range(K):
            for k in range(K):
                alpha = alpha - invC[j][k]*np.matmul(data[i,0:numOfDims],data[idx[k+1],0:numOfDims])
                beta = beta + invC[j][k]

        lambDa = alpha*1.0/beta
        for j in range(K):
            w[i][j] = 0
            for k in range(K):
                w[i][idx[j+1]] = w[i][idx[j+1]] + invC[j][k]*(np.matmul(data[i,0:numOfDims],data[idx[k+1],0:numOfDims])+lambDa)
        '''
        #another method
        # construct C
        for j in range(K):
            for k in range(K):
                C[j][k] = np.matmul(data[i,:] - data[idx[j + 1], :], data[i,:] - data[idx[k + 1], :])

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
'''
data = np.array([[1.5,1.5,0],[1,2,0],[2,1,0],[2,2,0],[1,1,0]])
result = constructW(data,4)
constructResult = np.matmul(result,data)

print(result)
print(constructResult)
'''
