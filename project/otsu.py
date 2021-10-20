
import cv2
import numpy as np
from abc import abstractmethod


class _BaseThresholding:

    @classmethod
    @abstractmethod
    def threshold(cls, img, **kwargs):
        """
        To find the threshold used for binarization of a grayscale image

        Parameters
        ----------
        img: np.array (height, width)
            The grayscale image

        Returns
        -------
        threshold: np.array (height, width)
            The threshold at each pixel position

        """
        raise NotImplementedError

    @classmethod
    def binarize(cls, img, **kwargs):
        """
        To binarize a grayscale image

        Parameters
        ----------
        img: np.array (height, width)
            The grayscale image

        Returns
        -------
        new_img: np.array (height, width)
            The binarized image

        """
        t = cls.threshold(img, **kwargs)
        return np.where(img >= t, 255, 0).astype(np.uint8)


class OTSU(_BaseThresholding):
    """
    Perform global OTSU thresholding
    """

    @classmethod
    def threshold(cls, img, **kwargs):
        histogram = cv2.calcHist(
            images=[img],
            channels=[0],
            mask=None,
            histSize=[256],
            ranges=[0, 256],
        )
        num_pixels = img.size
        probabilities = histogram.flatten() / num_pixels
        gray_levels = np.arange(len(probabilities))
        mean = gray_levels * probabilities

        best_threshold = -1
        min_variance = float('inf')

        for t in range(1, len(histogram)):
            lower_probability = np.sum(probabilities[:t])
            upper_probability = np.sum(probabilities[t:])
            if lower_probability == 0 or upper_probability == 0:
                continue

            lower_mean = np.sum(mean[:t]) / lower_probability
            upper_mean = np.sum(mean[t:]) / upper_probability

            lower_variance = np.sum(
                ((gray_levels[:t] - lower_mean) ** 2) * probabilities[:t]) / lower_probability
            upper_variance = np.sum(
                ((gray_levels[t:] - upper_mean) ** 2) * probabilities[t:]) / upper_probability

            variance = lower_probability * lower_variance + upper_probability * upper_variance
            if variance < min_variance:
                min_variance = variance
                best_threshold = t

        return np.full_like(img, fill_value=best_threshold)


class AdaptiveMeanThresholding(_BaseThresholding):
    """
    Threshold for each pixel is the mean of its (K x K) neighbourhood
    """

    @classmethod
    def threshold(cls, img, C=0, ksize=5, **kwargs):
        mean = cv2.blur(img, ksize=(ksize, ksize))
        t = np.where(mean > C, mean - C, 0).astype(np.uint8)
        return t
