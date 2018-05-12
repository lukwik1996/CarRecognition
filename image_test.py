import numpy as np
import os
from skimage import io, transform
from skimage.feature import canny
import cv2
from matplotlib import pyplot as plt


if __name__ == '__main__':

    path = 'D:\\si\\mini_test\\test\\'
    save = 'C:\\Users\\Lukasz\\PycharmProjects\\CarRecognition\\test_images\\test.jpg'
    img_size = 50

    for folder in os.listdir(path):
        for image in os.listdir(path+folder+"\\"):
            img = path+folder+"\\"+image
            photo = cv2.imread(img, 0)
            img = io.imread(img, as_grey=True)

            img = img / 255.0

            height, width = img.shape[0], img.shape[1]

            if height > width:
                img = transform.rescale(img, img_size / width)
                height, width = img.shape[0], img.shape[1]
                delta = abs(height - img_size) // 2
                img = img[delta:(img_size + delta), 0:(img_size + 10)]

            else:
                img = transform.rescale(img, img_size / height)
                height, width = img.shape[0], img.shape[1]
                delta = abs(width - img_size) // 2
                img = img[0:img_size, (delta - 5):(img_size + delta + 5)]

                edges = cv2.Canny(photo, 100, 150)

                plt.plot(122), plt.imshow(edges, cmap='gray')
                plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
                plt.show()
                cv2.waitKey(0)

                # cv2.imshow('image', img)
                # print(img.shape)
                # cv2.waitKey(0)

            if 1 == 1:
                break
        if 1 == 1:
            break
