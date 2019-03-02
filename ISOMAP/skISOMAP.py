import numpy as np
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from readYale import readYale
import os

filePath = '../YaleFace Data/CroppedYale'
data = readYale(filePath)
data = np.double(data)

if os.path.isfile('resultSk.npy'):
    result = np.load('resultSk.npy')
else:
    embedding = Isomap(n_components=2)
    result = embedding.fit_transform(data[:,0:data.shape[1]-1])
    np.save('resultSk.npy', result)

for i in range(39):
    plt.scatter(result[20*i:20*i+20,0],result[20*i:20*i+20,1])
plt.show()
