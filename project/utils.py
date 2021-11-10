import os
import cv2


def read_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def save_image(img, path):
    dirname = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    cv2.imwrite(path, img)
