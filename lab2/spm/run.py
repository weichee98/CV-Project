import os
import numpy as np
from sklearn.svm import LinearSVC

from spm import SpatialPyramidMatching


def split_category_train_test(X, category, n_train=30, max_n_test=50):
    if len(X) < n_train:
        raise Exception("Number of images for category '{}' < n_train".format(category))
    np.random.seed(0)
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    train_idx = idx[:n_train]
    test_idx = idx[n_train:n_train + max_n_test]
    X = np.array(X)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = np.array([category] * len(X_train))
    y_test = np.array([category] * len(X_test))
    return X_train, X_test, y_train, y_test


def load_data(data_dir: str):
    categories = os.listdir(data_dir)
    X_train, X_test = [], []
    y_train, y_test = [], []
    for cat in categories:
        files = os.listdir(os.path.join(data_dir, cat))
        X_cat = [os.path.join(data_dir, cat, file) for file in files]
        X_train_cat, X_test_cat, y_train_cat, y_test_cat = split_category_train_test(X_cat, cat)
        X_train.append(X_train_cat)
        X_test.append(X_test_cat)
        y_train.append(y_train_cat)
        y_test.append(y_test_cat)
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)

    all_cat, train_counts = np.unique(y_train, return_counts=True)
    all_cat2, test_counts = np.unique(y_test, return_counts=True)
    assert(np.all(all_cat == all_cat2))

    n = len(max(all_cat, key=len))
    print("Category Counts")
    print("===============")
    for cat, train_count, test_count in zip(all_cat, train_counts, test_counts):
        print("{}:\ttrain: {}\ttest: {}".format(
            cat + " " * (n - len(cat)), train_count, test_count
        ))
    print()
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    data_dir = "data"
    X_train, X_test, y_train, y_test = load_data(data_dir)

    print("Fit Spatial Pyramid Matching")
    SPM = SpatialPyramidMatching(n_jobs=-1)
    SPM.fit(X_train)
    print()

    for L in range(4):
        print("Train LinearSVC (L = {})".format(L))
        X_train_spm = SPM.transform(X_train, L=L)
        model = LinearSVC(random_state=0)
        model.fit(X_train_spm, y_train)

        X_test_spm = SPM.transform(X_test, L=L)
        accuracy = model.score(X_test_spm, y_test)
        print("Level {}:\t Accuracy: {}\n".format(L, accuracy))
