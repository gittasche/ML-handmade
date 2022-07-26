import numpy as np
from numpy.typing import ArrayLike

from mlhandmade.preprocessing.data_preprocessiong import data_shuffle
from ..utils.validations import check_random_state

class KFoldCV:
    """
    k-fold cross-validation.
    This class allows to iterate over
    train and test indices for each cross
    validation fold

    Parameters
    ----------
    n_splits : int (default: 5)
        number of splits in each fold
    shuffle : bool (default: False)
        shuffle data before iterations
    random_state : int (default: 0)
        seed for debugging
    """
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int = 0
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.rgen = check_random_state(random_state)

    def _validate_params(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        
        if X.ndim < 2:
            X = np.atleast_2d(X)

        if y is not None:
            if not isinstance(y, np.ndarray):
                y = np.asarray(y)
            
            if y.ndim < 2:
                y = np.atleast_2d(y)

    def split(self, X: np.ndarray, y: np.ndarray = None):
        """
        Get iterator over all folds

        Parametrs
        ---------
        X : ndarray of shape (N, D)
            ndarray of input samples
        y : ndarray of shape (N,) (default: None)
            ndarray of input targets
        
        Yields
        ------
        train_index, test_index : ndarrays
            indices for train and test parts of folds
        """
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
            indices = data_shuffle(indices, rgen=self.rgen)
        
        fold_sizes = np.full(self.n_splits, self.num_samples_ // self.n_splits, dtype=int)
        fold_sizes[:self.num_samples_ % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop

def cross_val_score(
    estimator,
    X,
    y,
    *,
    score=None,
    cv=5,
    shuffle=False,
    random_state=0,
    **score_kwargs
):
    if score is None:
        raise ValueError("Choose availible score.")

    scores = np.zeros(cv)
    kf = KFoldCV(n_splits=cv, shuffle=shuffle, random_state=random_state)
    for score_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        estimator.fit(X[train_idx], y[train_idx])
        y_pred = estimator.predict(X[test_idx])
        scores[score_idx] = score(y[test_idx], y_pred, **score_kwargs)
    return np.mean(scores)