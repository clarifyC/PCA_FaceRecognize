import numpy as np
import os
from PIL import Image

homePath = r'E:\pycharm files\基于PCA的人脸识别\\'

def CreateDatabase(TrainDatabase):
    ############# File management
    TrainFiles = os.listdir(homePath+TrainDatabase)
    ############# Construction of 2D matrix from 1D image vectors
    T = []
    for i in TrainFiles:
        if i!='Thumbs.db':
            # Train_Number = Train_Number + 1 # Number of all images in the training database
            str_ = homePath+ TrainDatabase + i
            image = Image.open(str_)
            img = list(image.getdata())
            T.append(img)
    T = np.array(T).T
    # print(T)
    return T