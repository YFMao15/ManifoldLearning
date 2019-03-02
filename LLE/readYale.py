import numpy as np
import os
import cv2

def readYale(filePath, classNum, classSize):
    #get folder list
    files = os.listdir(filePath)
    SizeOfPic = 168*192*3
    NumOfSamples = classNum*classSize
    data = np.zeros((NumOfSamples, SizeOfPic+1),np.uint8)
    print(len(files))
    idx = 0
    classesLabel = 0
    NumAbandoned = 0
    #Read pics into data
    for subfolder in files:
        print(subfolder)
        numController = 0
        for file in os.listdir(os.path.join(filePath,subfolder)):
            if file.endswith(".pgm"):
                file = os.path.join(filePath,subfolder,file)
                img = cv2.imread(file)
                if img.size == SizeOfPic:
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
        if classesLabel == classNum:
            break
    #display one of the pics
    '''
    IdxOfPic = 400
    cv2.imshow('One example',data[IdxOfPic,0:SizeOfPic].reshape(192,168,3))
    print(data[IdxOfPic][SizeOfPic])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    return data

