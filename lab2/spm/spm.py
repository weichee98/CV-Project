import cv2
import numpy as np
from tqdm import tqdm
from faiss import Kmeans
from scipy.spatial import KDTree
from joblib import Parallel, delayed


class SpatialPyramidMatching:

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
        self._centroid = None

    @staticmethod
    def _extract_sift_features(path):
        sift = cv2.SIFT_create()
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        kp, description = sift.detectAndCompute(img, None)
        coordinate = np.array([it.pt for it in kp]).reshape((-1, 2))
        size = np.array([[img.shape[1], img.shape[0]]])
        return description, coordinate / size

    @staticmethod
    def _get_grid(level, x, y):
        grid_width = 1 << level
        col_grid = int(x * grid_width)
        row_grid = int(y * grid_width)
        return row_grid * grid_width + col_grid

    def extract_sift_features(self, paths):
        """

        Parameters
        ----------
        paths: array-like (num_paths,)
            A list of path strings to the image files

        Returns
        -------
        descriptions: np.array (num_image,)
            A list of descriptions for every image specified by ``paths``,
            where each description is a ``np.array`` with shape
            (num_keypoints, 128)
        coordinates: np.array (num_image,)
            A list of keypoint coordinates for every image specified by
            ``paths``, where each keypoint coordinate is a ``np.array``
            with shape (num_keypoints, 2)

        """
        sift_features = Parallel(n_jobs=self.n_jobs)(
            delayed(self._extract_sift_features)(path)
            for path in tqdm(
                paths, desc="Extract SIFT Features",
                ncols=100, total=len(paths), leave=False
            )
        )
        descriptions, coordinates = list(zip(*sift_features))
        return np.array(descriptions, dtype=object), np.array(coordinates, dtype=object)

    def fit(self, descriptions):
        """

        Parameters
        ----------
        descriptions: np.array (num_image,)
            A list of descriptions for every image specified by ``paths``,
            where each description is a ``np.array`` with shape
            (num_keypoints, 128)

        Returns
        -------
        self

        """
        features = [desc for desc in descriptions if desc is not None]
        features = np.concatenate(features)
        print("Clustering Training Data Shape: {}".format(features.shape))
        clustering = Kmeans(d=features.shape[1], k=self.K, niter=300, verbose=True, seed=0)
        clustering.train(features)
        self._centroid = KDTree(clustering.centroids)
        return self

    def _calculate_image_features(self, description, coordinate, L):
        """

        Parameters
        ----------
        description: np.array (num_keypoints, 128)
            The SIFT feature vectors of a single image
        coordinate: np.array (num_keypoints, 2)
            The coordinates for each SIFT vector of a single image
        L: int
            Current processing level

        Returns
        -------
        feature_vec: np.array
            The resultant feature vector after processed

        """
        dim = self.K * (-1 + 4 ** (L + 1)) // 3
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
                    grid = self._get_grid(l, x, y)
                    hist[grid] += 1
                hist = [it * w for it in hist]
                feature_vec += hist

        feature_vec = [it / (((self.K / 200) * 25) * (1 << L)) for it in feature_vec]
        return feature_vec

    def transform(self, descriptions, coordinates, L=None):
        """

        Parameters
        ----------
        descriptions: np.array (num_image,)
            A list of descriptions for every image specified by ``paths``,
            where each description is a ``np.array`` with shape
            (num_keypoints, 128)
        coordinates: np.array (num_image,)
            A list of keypoint coordinates for every image specified by
            ``paths``, where each keypoint coordinate is a ``np.array``
            with shape (num_keypoints, 2)
        L: int or None
            Number of levels, if ``None``, use the default ``L`` specified
            when creating the instance

        Returns
        -------
        features: np.array (num_image, num_features)
            The processed feature vectors for each image

        """
        if L is None:
            L = self.L
        if len(descriptions) != len(coordinates):
            raise ValueError(
                "len(descriptions) != len(coordinates): {} != {}"
                .format(len(descriptions), len(coordinates))
            )
        features = Parallel(n_jobs=self.n_jobs)(
            delayed(self._calculate_image_features)(desc, coord, L)
            for (desc, coord) in tqdm(
                zip(descriptions, coordinates), desc="Calculate Feature Vector",
                ncols=100, total=len(coordinates), leave=False
            )
        )
        return np.array(features)
