import numpy as np

def get_split_mask(X, feature, value):
    left_mask = X[:, feature] < value
    right_mask = X[:, feature] >= value
    return left_mask, right_mask
    
def find_splits(X):
    X_uniq = np.unique(X)
    return (X_uniq[1:] + X_uniq[:-1]) / 2

def gb_criterion(loss, residuals, y_true, y_pred, left_mask, right_mask):
    left = loss.gain(residuals[left_mask], y_true[left_mask], y_pred[left_mask])
    right = loss.gain(residuals[right_mask], y_true[right_mask], y_pred[right_mask])
    return -left - right

def get_best_split(loss, X, residuals, y_true, y_pred, features):
    min_gain = np.inf
    best_feature = None
    best_value = None

    for feature in features:
        split_values = find_splits(X[:, feature])
        for value in split_values:
            left_mask, right_mask = get_split_mask(X, feature, value)

            gain = gb_criterion(loss, residuals, y_true, y_pred, left_mask, right_mask)

            if gain < min_gain:
                min_gain = gain
                best_feature = feature
                best_value = value

    return min_gain, best_feature, best_value