from CreateDatabase import CreateDatabase
from Recognition import Recognition
from EigenfaceCore import EigenfaceCore
from PIL import Image
import cv2

def main():

    test_image_name = input('Enter test image name (a number between 1 to 10): ')

    test_image_path = 'TestDatabase/' + test_image_name + '.jpg'

    T = CreateDatabase('TrainDatabase')
    m, centered_data, eigenfaces = EigenfaceCore(T)
    output_name = Recognition(test_image_path, m, centered_data, eigenfaces)

    selected_image_path = '/data/Disk_A/biancongcong/test/Face/TrainDatabase/' + output_name
    print(selected_image_path)
    selected_image = Image.open(selected_image_path)

    test_image = Image.open(test_image_path)
    test_image.show()
    selected_image.show()

    print('Matched image is:', output_name)

    # test_image_name = input('Enter test image name (a number between 1 to 10): ')
    # test_image_path = 'TestDatabase/' + test_image_name + '.jpg'
    # im = cv2.imread(test_image_path)
    # T= CreateDatabase('TrainDatabase')
    # m, centered_data, eigenfaces = EigenfaceCore(T)
    # # print(m,'data', centered_data,'face', eigenfaces)
    # output_name = Recognition(test_image_path, m, centered_data, eigenfaces)
    #
    # selected_image_path = 'TrainDatabase/' + output_name
    # selected_image = cv2.imread(selected_image_path)
    #
    # cv2.imshow('Test Image', im)
    # cv2.imshow('Equivalent Image', selected_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print('Matched image is:', output_name)

if __name__ == '__main__':
    main()