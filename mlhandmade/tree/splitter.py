import numpy as np

def get_split_mask(X, feature, value):
    left_mask = X[:, feature] < value
    right_mask = X[:, feature] >= value
    return left_mask, right_mask

def split(x, y, value):
    left_mask = x < value
    right_mask = x >= value
    return y[left_mask], y[right_mask]

def split_dataset(X, y, feature, value):
    left_mask, right_mask = get_split_mask(X, feature, value)
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

def find_splits(X):
    X_uniq = np.unique(X)
    return (X_uniq[1:] + X_uniq[:-1]) / 2

def get_best_split(criterion, X, y, features):
    min_gain = np.inf
    best_feature = None
    best_value = None

    for feature in features:
        split_values = find_splits(X[:, feature])
        for value in split_values:
            splits = split(X[:, feature], y, value)
            gain = criterion.gain(y, splits)

            if gain < min_gain:
                min_gain = gain
                best_feature = feature
                best_value = value

    return min_gain, best_feature, best_value