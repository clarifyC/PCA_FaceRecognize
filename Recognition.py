import numpy as np
from PIL import Image
from sklearn.metrics import pairwise_distances
homePath = r'E:\pycharm files\基于PCA的人脸识别\\'
def Recognition(TestImage, m, A, eigenfaces):
    Train_Number = eigenfaces.shape[1]
    # 将二十张训练图（centered image）都映射到19维的特征空间中
    ProjectedImages = np.matmul(eigenfaces.T, A)
    # 读取测试图片
    image = Image.open(homePath + TestImage)
    InImage = np.array(image.getdata()).reshape(-1, 1)
    # 归一化/中心化
    Difference = InImage - m

    # 将测试图片（centered image）映射到19维的特征空间中
    ProjectedTestImage = np.matmul(eigenfaces.T, Difference)

    # 计算测试特征图到每一张训练特征图欧氏距离
    Euc_dist = []
    for i in range(Train_Number):
        q = ProjectedImages[:, i].reshape(1, -1)
        twoPoints = np.vstack((ProjectedTestImage.T, q))
        temp = pairwise_distances(twoPoints, metric='euclidean')[1, 0]
        Euc_dist.append(temp)
    out_idx = np.argmin(np.array(Euc_dist)) + 1
    return out_idx
