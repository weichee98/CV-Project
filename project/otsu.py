import cv2
import numpy as np
from abc import abstractmethod


class _BaseOTSU:

    @staticmethod
    @abstractmethod
    def threshold(img, **kwargs):
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
        kwargs: dict
            Keyword argument for ``threshold`` method

        Returns
        -------
        new_img: np.array (height, width)
            The binarized image

        """
        best_threshold = cls.threshold(img, **kwargs)
        return np.where(img >= best_threshold, 255, 0).astype(np.uint8)


class OTSU(_BaseOTSU):
    """
    Perform global thresholding
    """

    @staticmethod
    def threshold(img, **kwargs):
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


class GridOTSU(_BaseOTSU):
    """
    Split image into grids then perform thresholding on each grid
    """

    @staticmethod
    def threshold(img, num_col=1, num_row=1, var_threshold=0, default_threshold=0,
                  gauss_sigma=None, gauss_ksize=(5, 5)):
        if gauss_sigma is not None:
            img = cv2.GaussianBlur(img, ksize=gauss_ksize, sigmaX=gauss_sigma, sigmaY=gauss_sigma)

        threshold_mask = np.zeros_like(img)
        height, width = img.shape[:2]
        rows = np.linspace(0, height, num_row + 1).astype(int)
        cols = np.linspace(0, width, num_col + 1).astype(int)
        global_var = np.var(img)

        for r in range(num_row):
            for c in range(num_col):
                grid = img[rows[r]:rows[r + 1], cols[c]:cols[c + 1]]
                if var_threshold > 0 and np.var(grid) < global_var * var_threshold:
                    threshold_mask[rows[r]:rows[r + 1], cols[c]:cols[c + 1]] = default_threshold
                    continue
                t = OTSU.threshold(grid)
                threshold_mask[rows[r]:rows[r + 1], cols[c]:cols[c + 1]] = t
        return threshold_mask


class SlidingOTSU(_BaseOTSU):
    """
    Find threshold using sliding window
    """

    @staticmethod
    def threshold(img, kernel_size=(5, 5), stride=(1, 1), var_threshold=0, default_threshold=0,
                  gauss_sigma=None, gauss_ksize=(5, 5)):
        if gauss_sigma is not None:
            img = cv2.GaussianBlur(img, ksize=gauss_ksize, sigmaX=gauss_sigma, sigmaY=gauss_sigma)

        threshold_mask = np.zeros_like(img, dtype=float)
        num_repetition = np.zeros_like(img)
        height, width = img.shape[:2]
        rows = np.linspace(0, (height - kernel_size[1]), (height - kernel_size[1]) // stride[1] + 1).astype(int)
        cols = np.linspace(0, (width - kernel_size[0]), (width - kernel_size[0]) // stride[0] + 1).astype(int)
        global_var = np.var(img)

        for r in range(len(rows)):
            for c in range(len(cols)):
                window = img[rows[r]:rows[r] + kernel_size[1], cols[c]:cols[c] + kernel_size[0]]
                if var_threshold > 0 and np.var(window) < global_var * var_threshold:
                    threshold_mask[rows[r]:rows[r] + kernel_size[1],
                    cols[c]:cols[c] + kernel_size[0]] += default_threshold
                else:
                    t = OTSU.threshold(window)
                    threshold_mask[rows[r]:rows[r] + kernel_size[1], cols[c]:cols[c] + kernel_size[0]] += t
                num_repetition[rows[r]:rows[r] + kernel_size[1], cols[c]:cols[c] + kernel_size[0]] += 1

        return threshold_mask / num_repetition
