import os
import numpy as np


class CaltechDataLoader:

    def __init__(self, n_train=30, max_n_test=50):
        self.n_train = n_train
        self.max_n_test = max_n_test

    def load_data(self, data_dir):
        n = self.n_train + self.max_n_test
        X, y = [], []
        categories = os.listdir(data_dir)
        for cat in categories:
            files = os.listdir(os.path.join(data_dir, cat))
            if len(files) < self.n_train:
                raise Exception("Number of images for category '{}' < n_train".format(cat))
            X_cat = [os.path.join(data_dir, cat, file) for file in files[:n]]
            X += X_cat
            y += [cat] * len(X_cat)
        return np.array(X), np.array(y)

    def split_category_train_test(self, y, seed=0):
        train_idx = []
        test_idx = []

        for cat in np.unique(y):
            idx = np.argwhere(y == cat).flatten()
            np.random.seed(seed)
            np.random.shuffle(idx)
            cat_train_idx = idx[:self.n_train]
            cat_test_idx = idx[self.n_train:self.n_train + self.max_n_test]
            train_idx.append(cat_train_idx)
            test_idx.append(cat_test_idx)

        train_idx = np.concatenate(train_idx)
        test_idx = np.concatenate(test_idx)
        return train_idx, test_idx
