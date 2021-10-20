import os
import re
import cv2
import argparse

import numpy as np
import pytesseract
from Levenshtein import ratio
from otsu import OTSU, AdaptiveMeanThresholding
from utils import read_image, save_image


def ocr(img, ref_text=None):
    text = pytesseract.image_to_string(img)
    text = re.sub(r"(\n\s*){2}(\s+)", "\n\n", text)
    if ref_text is None:
        dist = None
    else:
        dist = ratio(ref_text, text)
    return text, dist


def plot_text(img, text, dist=None, font_size=0.6):
    margin = 20
    line_margin = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1
    text_im = np.ones_like(img) * 255
    text = text.split("\n")
    x, y = margin, margin
    for line in text:
        text_size = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += text_size[1] + line_margin
        cv2.putText(text_im, line, (x, y), font, font_size, 0, font_thickness, lineType=cv2.LINE_AA)

    if dist is not None:
        line = "Accuracy: {:.5f}".format(dist)
        text_size = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        x = img.shape[1] - margin - text_size[0]
        y = img.shape[0] - margin
        cv2.putText(text_im, line, (x, y), font, font_size, 0, font_thickness, lineType=cv2.LINE_AA)
    return text_im


def plot_sample(img, text, dist, output_path, thresh_map=None, concat="vertical"):
    text_im = plot_text(img, text, dist, font_size=0.5)
    if thresh_map is None:
        images = [img, text_im]
    else:
        images = [thresh_map, img, text_im]
    if concat == "vertical":
        out_im = cv2.vconcat(images)
    elif concat == "horizontal":
        out_im = cv2.hconcat(images)
    else:
        raise NotImplementedError("invalid concat option: {}".format(concat))
    save_image(out_im, output_path)


def sample01(data_dir, output_dir):
    img_name = "sample01"
    file = "{}.png".format(img_name)
    img_path = os.path.join(data_dir, file)
    img = read_image(img_path)

    with open(os.path.join(data_dir, '{}.txt'.format(img_name)), 'r') as f:
        ref = f.read()

    # Original Image
    text, dist = ocr(img, ref)
    plot_sample(img, text, dist, os.path.join(output_dir, "{}_none.png".format(img_name)), concat="vertical")

    # OTSU
    otsu_map = OTSU.threshold(img)
    otsu = OTSU.binarize(img)
    text, dist = ocr(otsu, ref)
    plot_sample(otsu, text, dist, os.path.join(output_dir, "{}_otsu.png".format(img_name)),
                thresh_map=otsu_map, concat="vertical")

    # Adaptive Mean Thresholding
    mean_map = AdaptiveMeanThresholding.threshold(img, kernel_size=3, C=2)
    mean = AdaptiveMeanThresholding.binarize(img, kernel_size=3, C=2)
    mean = cv2.medianBlur(mean, 3)
    text, dist = ocr(mean, ref)
    plot_sample(mean, text, dist, os.path.join(output_dir, "{}_mean.png".format(img_name)),
                thresh_map=mean_map, concat="vertical")

    # Gaussian Blurring + Adaptive Mean Thresholding
    gauss = cv2.GaussianBlur(img, (5, 5), sigmaX=1., sigmaY=1.)
    gauss_map = AdaptiveMeanThresholding.threshold(gauss, kernel_size=3, C=2)
    gauss = AdaptiveMeanThresholding.binarize(gauss, kernel_size=3, C=2)
    text, dist = ocr(gauss, ref)
    plot_sample(gauss, text, dist, os.path.join(output_dir, "{}_gauss.png".format(img_name)),
                thresh_map=gauss_map, concat="vertical")


def sample02(data_dir, output_dir):
    img_name = "sample02"
    file = "{}.png".format(img_name)
    img_path = os.path.join(data_dir, file)
    img = read_image(img_path)

    with open(os.path.join(data_dir, '{}.txt'.format(img_name)), 'r') as f:
        ref = f.read()

    # Original Image
    text, dist = ocr(img, ref)
    plot_sample(img, text, dist, os.path.join(output_dir, "{}_none.png".format(img_name)), concat="horizontal")

    # OTSU
    otsu_map = OTSU.threshold(img)
    otsu = OTSU.binarize(img)
    text, dist = ocr(otsu, ref)
    plot_sample(otsu, text, dist, os.path.join(output_dir, "{}_otsu.png".format(img_name)),
                thresh_map=otsu_map, concat="horizontal")

    # Adaptive Mean Thresholding
    mean_map = AdaptiveMeanThresholding.threshold(img, kernel_size=3, C=1)
    mean = AdaptiveMeanThresholding.binarize(img, kernel_size=3, C=1)
    mean = cv2.medianBlur(mean, 3)
    text, dist = ocr(mean, ref)
    plot_sample(mean, text, dist, os.path.join(output_dir, "{}_mean.png".format(img_name)),
                thresh_map=mean_map, concat="horizontal")

    # Gaussian Blurring + Adaptive Mean Thresholding
    gauss = cv2.GaussianBlur(img, (5, 5), sigmaX=1., sigmaY=1.)
    gauss_map = AdaptiveMeanThresholding.threshold(gauss, kernel_size=3, C=1)
    gauss = AdaptiveMeanThresholding.binarize(gauss, kernel_size=3, C=1)
    text, dist = ocr(gauss, ref)
    plot_sample(gauss, text, dist, os.path.join(output_dir, "{}_gauss.png".format(img_name)),
                thresh_map=gauss_map, concat="horizontal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./images')
    parser.add_argument('--output', type=str, default='./results')
    args = parser.parse_args()

    data_dir = args.data
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sample01(data_dir, output_dir)
    sample02(data_dir, output_dir)
