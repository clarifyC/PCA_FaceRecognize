from PIL import Image

from CreateDatabase import CreateDatabase
from EigenfaceCore import EigenfaceCore
from Recognition import Recognition

homePath = r'E:\pycharm files\基于PCA的人脸识别\\'
TrainDatabase = r"TrainDatabase\\"
T = CreateDatabase(homePath+TrainDatabase)
m,A,Eigenfaces = EigenfaceCore(T)

# 输入测试图片
TestImage = input('Enter test image name (a number between 1 to 10): ')
TestImage = 'TestDatabase\\' + TestImage + '.jpg'
OutputName=Recognition(homePath+TestImage, m, A, Eigenfaces)

print("Matched image is:",str(OutputName)+".jpg")

# 绘制测试图片
image_test = Image.open(homePath+TestImage)
image_test.show()

# 绘制匹配到的图片
SelectedImage = homePath + r"TrainDatabase\\" +str(OutputName) + ".jpg"
image_select = Image.open(SelectedImage)
image_select.show()
