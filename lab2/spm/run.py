import os
import json
import argparse

from sklearn.svm import LinearSVC
from data import CaltechDataLoader
from spm import SpatialPyramidMatching

N_TRAIN = 30
N_TEST = 50
N_SEED = 10
N_LEVEL = 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--output', type=str, default='./')
    args = parser.parse_args()

    data_dir = args.data
    dataloader = CaltechDataLoader(n_train=N_TRAIN, max_n_test=N_TEST)
    X, y = dataloader.load_data(data_dir)

    SPM = SpatialPyramidMatching(n_jobs=-1)
    descriptions, coordinates = SPM.extract_sift_features(X)

    results = list()

    for seed in range(N_SEED):
        print("\nSeed: {}".format(seed))
        train_idx, test_idx = dataloader.split_category_train_test(y, seed)

        train_descriptions = descriptions[train_idx]
        test_descriptions = descriptions[test_idx]
        train_coordinates = coordinates[train_idx]
        test_coordinates = coordinates[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print("Fit Spatial Pyramid Matching")
        SPM.fit(train_descriptions)

        for L in range(N_LEVEL):
            print("Train LinearSVC (L = {})".format(L))
            X_train = SPM.transform(train_descriptions, train_coordinates, L=L)
            model = LinearSVC(random_state=0)
            model.fit(X_train, y_train)

            X_test = SPM.transform(test_descriptions, test_coordinates, L=L)
            accuracy = model.score(X_test, y_test)
            results.append({
                "seed": seed,
                "level": L,
                "accuracy": accuracy
            })
            print("Level {}:\t Accuracy: {}".format(L, accuracy))

    out_dir = args.output
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"results": results}, f)
