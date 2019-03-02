import numpy as np
import os
def generateMap(data):
    NumOfSamples = data.shape[0]
    NumOfDims = data.shape[1]
    # generate distance map
    #dataName = './OriginDisMap20.npy'
    dataName = 'OriginMap.npy'
    if os.path.isfile(dataName):
        originDisMap = np.load(dataName)
    else:
        originDisMap = np.zeros((NumOfSamples, NumOfSamples), dtype=np.float)
        for i in range(NumOfSamples):
            print(i + 1, "/", NumOfSamples)
            for j in range(NumOfSamples):
                originDisMap[i][j] = np.sqrt(np.sum((data[i, 0:NumOfDims] - data[j, 0:NumOfDims]) ** 2))
        np.save(dataName, originDisMap)


    return originDisMap
