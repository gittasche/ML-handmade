import numpy as np

def get_split_mask(X, feature, value):
    left_mask = X[:, feature] < value
    right_mask = X[:, feature] >= value
    return left_mask, right_mask

def find_splits(X):
    X_uniq = np.unique(X)
    return (X_uniq[1:] + X_uniq[:-1]) / 2

def get_best_split(criterion, X, y, features, sample_weight):
    min_gain = np.inf
    best_feature = None
    best_value = None

    for feature in features:
        split_values = find_splits(X[:, feature])
        for value in split_values:
            left_mask, right_mask = get_split_mask(X, feature, value)
            splits = (y[left_mask], y[right_mask])
            splits_weight = (sample_weight[left_mask], sample_weight[right_mask])

            if splits_weight[0].max() == 0.0 or splits_weight[1].max() == 0.0:
                continue

            gain = criterion.gain(y, splits, splits_weight)

            if gain < min_gain:
                min_gain = gain
                best_feature = feature
                best_value = value

    return min_gain, best_feature, best_value