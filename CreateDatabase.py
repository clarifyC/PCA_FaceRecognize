import numpy as np
import os
from PIL import Image

def CreateDatabase(TrainDatabase):
    ############# File management
    TrainFiles = os.listdir(TrainDatabase)
    Train_Number = 0;
    # 统计训练图片数量
    for i in range(len(TrainFiles)):
        if TrainFiles[i]!='Thumbs.db':
            Train_Number = Train_Number + 1
    T = []
    # 按顺序读取文件
    for i in range(1,Train_Number+1):
        str_ = TrainDatabase + str(i) + ".jpg"
        image = Image.open(str_)
        img = list(image.getdata())
        T.append(img)
    T = np.array(T).T
    return T