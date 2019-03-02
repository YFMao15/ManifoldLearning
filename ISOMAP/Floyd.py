import numpy as np
#Implement Floyd algorithm to find the shortest path
def floyd(disMap):
    N = disMap.shape[0]
    for k in range(N):
        print(k+1, '/', N)
        for i in range(N):
            for j in range(N):
                if disMap[i][j] > disMap[i][k] + disMap[k][j]:
                    disMap[i][j] = disMap[i][k] + disMap[k][j]

    return disMap

