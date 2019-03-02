import numpy as np
import matplotlib.pyplot as plt
from generateMap import generateMap
from constructW import constructW
from generateNY import generateNY
from readYale import readYale
import os
import scipy.io
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

filePath = '../YaleFace Data/CroppedYale'
K = 5 #Number of nearest neighbors
d = 10 #Dimension preserved, increase d to improve the accuracy
benchmark = False #Change to true to use lle method provided by sklearn
classNum = 38
classSize = 40

#Read data and generate map
data0 = readYale(filePath, classNum, classSize)
data = np.double(data0)
map = generateMap(data[:,0:data.shape[1]-1])

#generate reconstruction weight and the final result
if not benchmark:
    if os.path.isfile('w'+str(K)+'.npy'):
        w = np.load('w'+str(K)+'.npy')
    else:
        w = constructW(map,data[:,0:data.shape[1]-1],K)
        np.save('w'+str(K)+'.npy', w)
    result = generateNY(w,d)
else:
    #benchmark
    data0 = readYale(filePath, classNum, classSize)
    data = data0[:,0:data0.shape[1]-1]
    scipy.io.savemat('data.mat', dict(x=data,y=0))
    np.save('data.npy',data)
    print('Done')
    lle = LocallyLinearEmbedding(n_components=2)
    result = lle.fit_transform(data[:,0:data.shape[1]-1])
#print(result)

colors = ['#330000','#CC6600','#FFFF66','#CCFF99','#FFCCCC','#99FFCC','#CCFFFF','#99CCFF',
    '#6666FF','#9933FF','#FF33FF','#FF3399','#A0A0A0','#99004C','#330066','#0080FF', 
    '#4C0099','#990099','#FFFFCC','#99FF99','#0000FF','#CC00CC','#FF007F','#E0E0E0',
    '#FFFF00','#99FF33','#33FF99','#B266FF','#FF99CC','#808080','#000000','#330019',
    '#000033','#003319','#193300','#331900','#9999FF','#330033']

for i in range(classNum):
    plt.scatter(result[classSize*i:classSize*(i+1), 0], result[classSize*i:classSize*(i+1), 1], color = classSize*[colors[i]])
plt.show()


#Divide the data into training set and test set
trainSize = 30
testSize = 10
labels = data0[:,data0.shape[1]-1]
dataCSet = result
dataTrain = np.zeros((classNum*trainSize, d))
dataTest = np.zeros((classNum*testSize, d))
dataTrainLabel = np.zeros(classNum*trainSize)
dataTestLabel = np.zeros(classNum*testSize)
for i in range(classNum):
    for j in range(trainSize):
        dataTrain[trainSize*i + j, :] = dataCSet[classSize*i + j, :]
        dataTrainLabel[trainSize*i + j] = labels[classSize*i + j]
    for j in range(testSize):
        dataTest[testSize*i + j, :] = dataCSet[classSize*i + trainSize + j, :]
        dataTestLabel[testSize*i + j] = labels[classSize*i + trainSize + j]

#Use KNN to classify
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(dataTrain, dataTrainLabel)
predictTrainLabel = neigh.predict(dataTrain)
err = 1.0*np.count_nonzero(dataTrainLabel - predictTrainLabel)/np.shape(dataTrainLabel)[0]
print("Accuracy of training")
print(1-err)

predictTestLabel = neigh.predict(dataTest)
err = 1.0*np.count_nonzero(dataTestLabel - predictTestLabel)/np.shape(dataTestLabel)[0]
print("Accuracy of testing")
print(1-err)

