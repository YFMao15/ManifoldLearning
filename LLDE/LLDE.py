import os
import numpy as np
from constructW import constructW
from readYale import readYale
from scipy.linalg import eig, lu
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

filePath = '../YaleFace Data/CroppedYale'
distanceType = 0#0:Euclidean distance, 1:Normalized euclidean distance, 2:Cosine distance
K = 5 #Number of neighbors
d = 100 #dimension preserved
classNum = 38
classSize = 40
trainSize = 30
testSize = 10
pcaDim = 89
#Step0 read data and PCA
data = readYale(filePath, classNum, classSize)
dataTrainX = np.zeros((classNum*trainSize, data.shape[1]-1))
dataTestX = np.zeros((classNum*testSize, data.shape[1]-1))
for i in range(classNum):
    for j in range(trainSize):
        dataTrainX[trainSize*i+j, :] = data[classSize*i + j,0:data.shape[1]-1]
    for j in range(testSize):
        dataTestX[testSize * i + j, :] = data[classSize * i + j + trainSize, 0:data.shape[1]-1]
pca = PCA(n_components=pcaDim)
pca.fit(dataTrainX)
dataTrainXPCA = pca.transform(dataTrainX)
X = dataTrainXPCA.T

dataTestXPCA = pca.transform(dataTestX)
XTest = dataTestXPCA.T


#Step1 generate reconstruction weight
wName = 'w' + str(K) + '.' + str(distanceType) + '.npy'
if os.path.isfile(wName):
    w = np.load(wName)
else:
    w = constructW(dataTrainXPCA, K, distanceType)
    np.save(wName, w)
print('Weight reconstructed!')

#Step2 construct matrix M
I = np.eye(w.shape[0])
M = np.matmul((I-w).T,I-w)
print('M constructed!')

#Step3 construct matrix XMX^T
XMXT = np.matmul(np.matmul(X,M),X.T)
print('XMXT generated!')

#Step4 generate Sb, Sw and Sb - muSw
mu = 0.001
x_mean = np.transpose([np.mean(X, 1)])
Sb = np.zeros((x_mean.size, x_mean.size))
Sw = np.zeros((x_mean.size, x_mean.size))

for i in range(classNum):
    print(i,'/',classNum)
    xi_mean = np.transpose([np.mean(X[:,trainSize*i:trainSize*(i+1)],1)])
    Sb += trainSize*np.matmul(xi_mean - x_mean, (xi_mean - x_mean).T)
    for j in range(trainSize):
        Sw += np.matmul(X[:, i*trainSize + j] - xi_mean, (X[:, i*trainSize + j] - xi_mean).T)

Sbmw = Sb - mu*Sw
print('Sbmw generated!')

#Step5 generate V and Y
eigs, eigV = eig(XMXT-Sbmw, np.matmul(X, X.T))
eigIdx = eigs.argsort()
print(eigs)
V = eigV[:, eigIdx[0:d]]
Y = np.matmul(V.T, X)
print('Dimension reduced!')

#Visualize the result

colors = ['#330000','#CC6600','#FFFF66','#CCFF99','#FFCCCC','#99FFCC','#CCFFFF','#99CCFF',
    '#6666FF','#9933FF','#FF33FF','#FF3399','#A0A0A0','#99004C','#330066','#0080FF',
    '#4C0099','#990099','#FFFFCC','#99FF99','#0000FF','#CC00CC','#FF007F','#E0E0E0',
    '#FFFF00','#99FF33','#33FF99','#B266FF','#FF99CC','#808080','#000000','#330019',
    '#000033','#003319','#193300','#331900','#9999FF','#330033']
for i in range(38):
    plt.scatter(Y[0,classSize*i:classSize*(i+1)],Y[1,classSize*i:classSize*(i+1)], color=classSize*[colors[i]])

plt.show()


print('Visualized!')

#Reduce the test set
YTest = np.matmul(V.T, XTest)

#Use KNN to classify
XTrainLabel = np.zeros(classNum*trainSize)
XTestLabel = np.zeros(classNum*testSize)
for i in range(classNum):
    for j in range(trainSize):
        XTrainLabel[trainSize*i + j] = i
    for j in range(testSize):
        XTestLabel[testSize*i + j] = i

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(Y.T, XTrainLabel)
yPre = neigh.predict(Y.T)
err = np.count_nonzero(yPre - XTrainLabel)/yPre.size
print("Accuracy on training set")
print(1-err)

yPreT = neigh.predict(YTest.T)
err = np.count_nonzero(yPreT - XTestLabel)/yPreT.size
print("Accuracy on testing set")
print(1-err)




