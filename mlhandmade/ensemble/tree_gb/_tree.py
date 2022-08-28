import numpy as np

from ...utils.validations import check_random_state, _num_samples, _num_features
from ._splitter import get_best_split, get_split_mask

class GBTree:
    def __init__(
        self,
        loss,
        max_features=None,
        min_samples_leaf=1,
        max_depth=3,
        random_state=0,
        n_samples=None,
        n_features=None
    ):
        self.loss = loss
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.rgen = check_random_state(random_state)

        self.n_samples = n_samples
        self.n_features = n_features
        self.is_leaf = False

    def fit(self, X, residuals, y, y_pred):
        self.n_samples = _num_samples(X)
        self.n_features = _num_features(X)
        self._build_tree(X, residuals, y, y_pred)

    def _build_tree(self, X, residuals, y_true, y_pred):
        try:
            assert self._check_leaf(self.n_samples)
            
            features = self.rgen.choice(self.n_features, self.max_features)
            gain, feature, value = get_best_split(
                self.loss,
                X,
                residuals,
                y_true,
                y_pred,
                features
            )
            assert gain != np.inf
            
            self.feature = feature
            self.value = value

            left_mask, right_mask = get_split_mask(X, feature, value)

            self.left = GBTree(
                self.loss,
                self.max_features,
                self.min_samples_leaf,
                self.max_depth - 1,
                self.rgen,
                self.n_samples,
                self.n_features
            )
            self.left._build_tree(
                X[left_mask],
                residuals[left_mask],
                y_true[left_mask],
                y_pred[left_mask]
            )

            self.right = GBTree(
                self.loss,
                self.max_features,
                self.min_samples_leaf,
                self.max_depth - 1,
                self.rgen,
                self.n_samples,
                self.n_features
            )
            self.right._build_tree(
                X[right_mask],
                residuals[right_mask],
                y_true[right_mask],
                y_pred[right_mask]
            )
        except AssertionError:
            self.is_leaf = True
            self.calculate_leaf_value(residuals, y_true, y_pred)

    def _check_leaf(self, num_samples_node):
        samples_condition = num_samples_node > self.min_samples_leaf
        depth_condition = self.max_depth > 0
        return samples_condition & depth_condition

    def calculate_leaf_value(self, residuals, y_true, y_pred):
        denom = self.loss.hess(y_true, y_pred).sum() + self.loss.regularization
        self.leaf_value = residuals.sum() / denom


    def predict_sample(self, x):
        if not self.is_leaf:
            if x[self.feature] < self.value:
                return self.left.predict_sample(x)
            else:
                return self.right.predict_sample(x)
        else:
            return self.leaf_value

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            y_pred[i] = self.predict_sample(x)
        return y_pred
