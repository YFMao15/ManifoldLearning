import numpy as np
import os
import cv2

def readYale(filePath, classNum, classSize):
    #get folder list
    files = os.listdir(filePath)
    SizeOfPic = 168*192*3
    NumOfSamples = classNum*classSize
    NewWidth = 100
    NewHeight = 100
    data = np.zeros((NumOfSamples, NewWidth*NewHeight+1),np.uint8)
    print(len(files))
    idx = 0
    classesLabel = 0
    NumAbandoned = 0
    #Read pics into data
    cnt = 0
    for subfolder in files:
        print(subfolder)
        cnt += 1
        if(cnt == classNum+1):
            break
        numController = 0
        for file in os.listdir(os.path.join(filePath,subfolder)):
            if file.endswith(".pgm"):
                file = os.path.join(filePath,subfolder,file)
                imgRgb = cv2.imread(file)

                if imgRgb.size == SizeOfPic:
                    imgGray = cv2.cvtColor(imgRgb, cv2.COLOR_RGB2GRAY)
                    img = cv2.resize(imgGray, (NewHeight, NewWidth))
                    data[idx,0:img.size] = img.reshape(1,img.size)
                    data[idx,img.size] = classesLabel
                    idx = idx + 1
                    numController = numController + 1
                    if numController == classSize:
                        break

                    #print(idx)
                else:
                    print(file)
                    print(NumAbandoned+1)
                    NumAbandoned = NumAbandoned+1
        classesLabel = classesLabel + 1
    #display one of the pics
    '''
    IdxOfPic = 400
    cv2.imshow('One example',data[IdxOfPic,0:SizeOfPic].reshape(192,168,3))
    print(data[IdxOfPic][SizeOfPic])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    return data

'''
a = readYale('../YaleFace Data/CroppedYale/')
print(a.shape)
'''