import cv2
import numpy as np
from PIL import Image

# def Recogntion(TestImage,m,A,Eigenfaces):
#     ProjectedImages = []
#     Train_Number =  Eigenfaces.shape[1]
#     for i in range(Train_Number):
#         temp =  Eigenfaces.T.dot(A[:,i])
#         ProjectedImages.append(temp)
#     InputImage=cv2.imread(TestImage)
#     temp = InputImage[:,:,0]
#     irow,icol = temp.shape
#     InImage = temp.reshape(irow * icol, 1)
#     Difference = np.double(InImage) - m
#     ProjectedTestImage = Eigenfaces.T
#     Euc_dist = []
#     for i in range(Train_Number):
#         q = ProjectedImages[i]
#         print(type(q))
#         q=q.reshape(19,1)
#         print(q.shape)
#         temp = np.linalg.norm(ProjectedTestImage - q) ** 2
#         Euc_dist.append(temp)
#
#     Euc_dist_min = min(Euc_dist)
#     Recognized_index = np.argmin(Euc_dist)
#     OutputName = str(Recognized_index) + '.jpg'
#
#     return OutputName
#


def Recognition(test_image, m, A, Eigenfaces):
    ProjectedImages = []
    Train_Number = Eigenfaces.shape[1]
    for i in range(Train_Number):
        temp = Eigenfaces.T.dot(A[:,i])
        ProjectedImages.append(temp)
    InputImage = Image.open(test_image).convert('L')
    temp = np.array(InputImage)
    irow, icol = temp.shape
    InImage = temp.reshape(irow*icol, 1)
    Difference = np.double(InImage) - m
    ProjectedTestImage = Eigenfaces.T
    Euc_dist = []
    for i in range(Train_Number):
        q = ProjectedImages[i].reshape(-1, 1)
        temp = np.linalg.norm(ProjectedTestImage - q)**2
        Euc_dist.append(temp)
    Euc_dist_min = min(Euc_dist)
    Recognized_index = np.argmin(Euc_dist)
    OutputName = str(Recognized_index) + '.jpg'
    return OutputName

