import numpy as np
from numpy.typing import ArrayLike
from mlhandmade.preprocessing.data_preprocessiong import data_shuffle

class KFoldCV:
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int = 0
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _validate_params(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if X.ndim < 2:
            X = np.atleast_2d(X)

        if y is not None:
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            
            if y.ndim < 2:
                y = np.atleast_2d(y)

    def split(self, X: np.ndarray, y: np.ndarray = None):
        self._validate_params(X, y)
        self.num_samples_ = X.shape[0]
        indices = np.arange(self.num_samples_)
        for test_index in self._iter_test_masks(X, y):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    def _iter_test_masks(self, X: np.ndarray, y: np.ndarray):
        for test_index in self._iter_test_indices(X, y):
            test_mask = np.zeros(self.num_samples_, dtype=bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X: np.ndarray, y: np.ndarray = None):
        indices = np.arange(self.num_samples_)
        if self.shuffle:
            indices = data_shuffle(indices)
        
        fold_sizes = np.full(self.n_splits, self.num_samples_ // self.n_splits, dtype=int)
        fold_sizes[:self.num_samples_ % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop