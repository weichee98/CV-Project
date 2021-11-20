import os
import cv2
import logging
import argparse

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from thresholding import OTSU, AdaptiveMeanThresholding, RegressionThresholding
from utils import read_image, ocr, plot_sample


def otsu_adaptive(img, ref, img_name, output_dir, orientation):
    # Original Image
    text, dist = ocr(img, ref)
    plot_sample(img, text, dist, os.path.join(output_dir, "{}_none.png".format(img_name)), concat=orientation)

    # OTSU
    otsu, otsu_map = OTSU.binarize(img)
    text, dist = ocr(otsu, ref)
    plot_sample(otsu, text, dist, os.path.join(output_dir, "{}_otsu.png".format(img_name)),
                thresh_map=otsu_map, concat=orientation)
    OTSU.plot_histogram(img, os.path.join(output_dir, "{}_otsu_hist.png".format(img_name)), img_name)

    # Adaptive Mean Thresholding
    mean, mean_map = AdaptiveMeanThresholding.binarize(img, kernel_size=3, C=2)
    mean = cv2.medianBlur(mean, 3)
    text, dist = ocr(mean, ref)
    plot_sample(mean, text, dist, os.path.join(output_dir, "{}_mean.png".format(img_name)),
                thresh_map=mean_map, concat=orientation)

    # Gaussian Blurring + Adaptive Mean Thresholding
    gauss = cv2.GaussianBlur(img, (5, 5), sigmaX=1., sigmaY=1.)
    gauss, gauss_map = AdaptiveMeanThresholding.binarize(gauss, kernel_size=3, C=2)
    text, dist = ocr(gauss, ref)
    plot_sample(gauss, text, dist, os.path.join(output_dir, "{}_gauss.png".format(img_name)),
                thresh_map=gauss_map, concat=orientation)


def regression(img, ref, img_name, output_dir, orientation):
    for downsample in [1, 2, 5, 10]:
        for P in [1, 2, 3]:
            # Linear Regression
            model = LinearRegression()
            linear, linear_map = RegressionThresholding.binarize(
                img, rgr_model=model, C=0.2, downsample=downsample, P=P
            )
            text, dist = ocr(linear, ref)
            plot_sample(
                linear, text, dist,
                os.path.join(output_dir, "{}_linear_down_{}_p_{}.png".format(img_name, downsample, P)),
                thresh_map=linear_map, concat=orientation
            )

            # MLP Regressor
            model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=10, random_state=0, verbose=True)
            mlp, mlp_map = RegressionThresholding.binarize(
                img, rgr_model=model, C=0.2, downsample=downsample, P=P
            )
            text, dist = ocr(mlp, ref)
            plot_sample(
                mlp, text, dist,
                os.path.join(output_dir, "{}_mlp_down_{}_p_{}.png".format(img_name, downsample, P)),
                thresh_map=mlp_map, concat=orientation
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./images')
    parser.add_argument('--output', type=str, default='./results/sample03')
    args = parser.parse_args()

    data_dir = args.data
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_name = "sample03"
    file = "{}.png".format(img_name)
    logging.basicConfig(
        filename=os.path.join(output_dir, "{}_log.txt".format(img_name)),
        filemode="w",
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%D %H:%M:%S"
    )

    img_path = os.path.join(data_dir, file)
    img = read_image(img_path)

    with open(os.path.join(data_dir, '{}.txt'.format(img_name)), 'r') as f:
        ref = f.read()

    otsu_adaptive(img, ref, img_name, output_dir, "horizontal")
    regression(img, ref, img_name, output_dir, "horizontal")
