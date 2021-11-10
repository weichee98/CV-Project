import os
import re
import cv2
import pytesseract
import numpy as np
from Levenshtein import ratio


def read_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def save_image(img, path):
    dirname = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    cv2.imwrite(path, img)


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