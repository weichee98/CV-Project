import cv2
import numpy as np
from tqdm import tqdm
from faiss import Kmeans
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed


class SpatialPyramidMatching(BaseEstimator, TransformerMixin):

    def __init__(self, L=0, K=200, n_jobs=1):
        """
        Parameters
        ----------
        L: int
            Number of levels
        K: int
            Number of clusters
        """
        self.L = L
        self.K = K
        self.n_jobs = n_jobs

    @staticmethod
    def _extract_sift_features(path):
        sift = cv2.SIFT_create()
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        kp, description = sift.detectAndCompute(img, None)
        coordinate = np.array([it.pt for it in kp])
        size = img.shape[1], img.shape[0]
        return description, coordinate, size

    @staticmethod
    def _get_grid(level, x, y, cols, rows):
        denom = 1 << level
        nx, ny = 0, 0
        for numer in range(denom):
            if (numer / denom) * cols <= x <= ((numer + 1) / denom) * cols:
                nx = numer
            if (numer / denom) * rows <= y <= ((numer + 1) / denom) * rows:
                ny = numer
        return ny * denom + nx

    def fit(self, X, y=None, **kwargs):
        """
        Parameters
        ----------
        X: np.array (num_samples,)
            An array-like object with elements as image file path strings
        y: any
            Not used
        kwargs: dict
            Not used

        Returns
        -------
        self

        """
        sift_desc = Parallel(n_jobs=self.n_jobs)(
            delayed(self._extract_sift_features)(path)
            for path in tqdm(X, desc="Extract SIFT Features", ncols=100)
        )
        features = [desc for (desc, _, _) in sift_desc if desc is not None]
        features = np.concatenate(features)
        print("Clustering Training Data Shape: {}".format(features.shape))
        clustering = Kmeans(d=features.shape[1], k=self.K, niter=300, verbose=True, seed=0)
        clustering.train(features)
        self._centroid = KDTree(clustering.centroids)
        return self

    def _calculate_image_features(self, path, L):
        dim = self.K * (-1 + 4 ** (L + 1)) // 3
        description, coordinate, size = self._extract_sift_features(path)
        if description is None:
            return [0] * dim

        feature_vec = []
        cluster = self._centroid.query(description)[1]
        for c in range(self.K):
            idx = np.argwhere(cluster == c).flatten()
            if len(idx) == 0:
                feature_vec += [0] * (dim // self.K)
                continue
            cluster_coordinates = coordinate[idx]
            for l in range(L + 1):
                w = 1 / (1 << L) if l == 0 else 1 / (1 << (L - l + 1))
                hist = [0] * (4 ** l)
                for (x, y) in cluster_coordinates:
                    grid = self._get_grid(l, x, y, size[0], size[1])
                    hist[grid] += 1
                hist = [it * w for it in hist]
                feature_vec += hist

        feature_vec = [it / (((self.K / 200) * 25) * (1 << L)) for it in feature_vec]
        return feature_vec

    def transform(self, X, L=None):
        """

        Parameters
        ----------
        X: np.array (num_samples,)
            An array-like object with elements as image file path strings
        L: int or None
            Number of levels

        Returns
        -------
        features: np.array (num_samples, num_features)

        """
        if L is None:
            L = self.L
        features = Parallel(n_jobs=self.n_jobs)(
            delayed(self._calculate_image_features)(path, L)
            for path in tqdm(X, desc="Extract Feature Vector", ncols=100)
        )
        return np.array(features)
