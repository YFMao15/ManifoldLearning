import numpy as np
import os
def generateMap(data, distanceClass):
    #distanceClass:0-euclidean distance, 1-normalized euclidean distance
    #2-cosine distance
    NumOfSamples = data.shape[0]
    NumOfDims = data.shape[1]
    Punish = 0
    # generate distance map
    dataName = 'originDisMap'+ str(distanceClass)+'.npy'
    if os.path.isfile(dataName):
        originDisMap = np.load(dataName)
    else:
        originDisMap = np.zeros((NumOfSamples, NumOfSamples), dtype=np.float)
        for i in range(NumOfSamples):
            print(i + 1, "/", NumOfSamples)
            for j in range(NumOfSamples):
                if distanceClass == 0:
                    originDisMap[i][j] = np.sqrt(np.sum((data[i, 0:NumOfDims] - data[j, 0:NumOfDims]) ** 2))
                elif distanceClass == 1:
                    originDisMap[i][j] = np.sqrt(np.sum((data[i, 0:NumOfDims]/np.linalg.norm(data[i, 0:NumOfDims]) - data[j, 0:NumOfDims]/np.linalg.norm(data[j, 0:NumOfDims])) ** 2))
                elif distanceClass == 2:
                    originDisMap[i][j] = abs(np.dot(data[i, 0:NumOfDims]/np.linalg.norm(data[i, 0:NumOfDims]), data[j, 0:NumOfDims]/np.linalg.norm(data[j, 0:NumOfDims])))*1000


            
        np.save(dataName, originDisMap)


    return originDisMap
