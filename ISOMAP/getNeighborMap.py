from readYale import readYale
import numpy as np
import os.path

def getNeighborMap(data, N, EpOrKNN, edition):
    #read Yale dataset
    NumOfSamples = data.shape[0]
    NumOfDims = data.shape[1]
    #generate distance map
    dataName = './OriginDisMap'+str(edition)+'.npy'
    if os.path.isfile(dataName):
        originDisMap = np.load(dataName)
    else:
        originDisMap = np.zeros((NumOfSamples,NumOfSamples),dtype=np.float)
        punish = 0
        for i in range(NumOfSamples):
            print(i+1,"/",NumOfSamples)
            for j in range(NumOfSamples):
                originDisMap[i][j] = np.sqrt(np.sum((data[i,0:NumOfDims] - data[j,0:NumOfDims])**2))

        np.save(dataName, originDisMap)

    #K nearest neighbors
    InfiniteDis = 10000000000000
    disMap = InfiniteDis*np.ones(originDisMap.shape, dtype=np.double)
    if EpOrKNN:
        #epsilon method
        mask = disMap > N
        disMap[mask] = InfiniteDis
        np.save("disMap20.npy", disMap)
    else:
        #KNN method
        for i in range(NumOfSamples):
            mask = disMap[i,:].argsort()[0:N+1]
            disMap[i,mask] = originDisMap[i,mask]
            disMap[mask,i] = originDisMap[mask,i]


    return disMap