import numpy as np
from itertools import combinations_with_replacement as combinations_w_r
from itertools import combinations
from typing import Union

from ..utils.validations import check_random_state

def data_shuffle(X: np.ndarray, y: np.ndarray = None, random_state: Union[int, None] = 0) -> np.ndarray:
    rgen = check_random_state(random_state)
    idx = np.arange(X.shape[0])
    rgen.shuffle(idx)
    if y is None:
        return X[idx]
    else:
        return X[idx], y[idx]

def ordinal(y: np.ndarray) -> np.ndarray:
    labels = np.unique(y)
    ordinal = np.zeros(y.shape, dtype=np.int64)
    for idx, cl in enumerate(labels):
        ordinal[y == cl] = idx
    return ordinal

def binary(y: np.ndarray, pos_label: int = 1, neg_label: int = -1) -> np.ndarray:
    labels = np.unique(y)
    if labels.shape[0] != 2:
        raise ValueError("Must be two classes to encode")
    return np.where(y == labels[0], pos_label, neg_label).astype(int)

def add_bias_feature(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        return np.insert(X, 0, 1)
    else:
        # np.hstack((np.ones((X.shape[0], 1)), X))
        return np.pad(X, [(0, 0), (1, 0)], mode="constant", constant_values=1)

def standardize(X: np.ndarray) -> np.ndarray:
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def onehot(y: np.ndarray) -> np.ndarray:
    n_values = np.max(y) + 1
    return np.eye(n_values, dtype=np.int64)[y]

def polynomial_features(X: np.ndarray, degree: int, interactions_only=False):
    n_samples, n_features = X.shape
    combinator = combinations if interactions_only else combinations_w_r
    combs = [combinator(range(n_features), i) for i in range(1, degree + 1)]
    product_tuples = [item for sublist in combs for item in sublist]
    n_output_features = len(product_tuples)

    XP = np.empty((n_samples, n_output_features))
    for i, idx_product in enumerate(product_tuples):
        XP[:, i] = np.prod(X[:, idx_product], axis=1)
    
    return XP