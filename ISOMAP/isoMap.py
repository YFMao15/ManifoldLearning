from getNeighborMap import getNeighborMap
from Floyd import floyd
from MDS import mds
import numpy as np
import os.path
import matplotlib.pyplot as plt
from readYale import readYale
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

edition = 1
#get neighbor map
filePath = '../YaleFace Data/CroppedYale'
data0 = readYale(filePath)
data = data0[:, 0:data0.shape[1]-1] #The last column of data0 is label
data = np.double(data)
pca = PCA(n_components=89)
#dataPCA = pca.fit_transform(data)
labels = data0[:,data0.shape[1]-1]
disMap = getNeighborMap(data, 4, False, edition)
print("Map generated!")

#get shortest path map
shortDisMapName = 'sMap'+str(edition)+'.npy'
if os.path.isfile(shortDisMapName):
    shortDisMap = np.load(shortDisMapName)
    print("Use pregenerated map")
else:
    shortDisMap = floyd(disMap)
    np.save(shortDisMapName, shortDisMap)
    print("shortest Path graph constructed!")

#MDS
result = mds(shortDisMap, 2)
print("Dimension Reduced!")
print(labels)

colors = ['#330000','#CC6600','#FFFF66','#CCFF99','#FFCCCC','#99FFCC','#CCFFFF','#99CCFF',
    '#6666FF','#9933FF','#FF33FF','#FF3399','#A0A0A0','#99004C','#330066','#0080FF',
    '#4C0099','#990099','#FFFFCC','#99FF99','#0000FF','#CC00CC','#FF007F','#E0E0E0',
    '#FFFF00','#99FF33','#33FF99','#B266FF','#FF99CC','#808080','#000000','#330019',
    '#000033','#003319','#193300','#331900','#9999FF','#330033']
for i in range(38):
    plt.scatter(result[20*i:20*(i+1),0],result[20*i:20*(i+1),1], color=20*[colors[i]])
plt.show()