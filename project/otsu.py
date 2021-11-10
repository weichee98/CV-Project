
import cv2
import numpy as np
from abc import abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


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
    def binarize(cls, img, return_threshold=True, **kwargs):
        """
        To binarize a grayscale image

        Parameters
        ----------
        img: np.array (height, width)
            The grayscale image
        return_threshold: bool
            Whether or not to return threshold map

        Returns
        -------
        new_img: np.array (height, width)
            The binarized image

        """
        t = cls.threshold(img, **kwargs)
        b = np.where(img >= t, 255, 0).astype(np.uint8)
        if return_threshold:
            return b, t
        return b


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


class RegressionThresholding(_BaseThresholding):
    """
    Thresholding using regression models
    """

    @classmethod
    def threshold(cls, img, rgr_model=LinearRegression(), C=0, downsample=None, **kwargs):
        if downsample is None or downsample <= 1:
            _height, _width = img.shape[0], img.shape[1]
            _img = img
        else:
            downsample = float(downsample)
            _width, _height = int(img.shape[1] / downsample), int(img.shape[0] / downsample)
            _img = cv2.resize(img, (_width, _height))

        x = np.repeat(np.expand_dims(np.arange(_width), axis=0), _height, axis=0)
        y = np.repeat(np.expand_dims(np.arange(_height), axis=1), _width, axis=1)
        data = np.stack([x, y, _img], axis=2).reshape((-1, 3))
        s = StandardScaler()
        data = s.fit_transform(data)
        X = data[:, [0, 1]]
        y = data[:, 2]

        rgr_model.fit(X, y)
        pred = rgr_model.predict(X) - C
        inv_pred = s.inverse_transform(np.stack([X[:, 0], X[:, 1], pred], axis=1))
        threshold = np.clip(inv_pred[:, 2], 0, 255)
        threshold = threshold.reshape(_img.shape).astype(np.uint8)

        if downsample is None or downsample <= 1:
            return threshold
        threshold = cv2.resize(threshold, (img.shape[1], img.shape[0]))
        return threshold




