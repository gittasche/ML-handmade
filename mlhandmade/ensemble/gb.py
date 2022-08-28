import numpy as np
import numbers

from ..base import BaseEstimator
from ..utils.validations import check_random_state
from .gb_losses import LeastSquaresLoss, LogisticLoss
from .tree_gb._tree import GBTree

LOSSES_DICT = {
    "log_loss": LogisticLoss,
    "squared_error": LeastSquaresLoss
}

class BaseGradientBoosting(BaseEstimator):
    def __init__(
        self,
        n_estimators,
        learning_rate,
        loss,
        max_features,
        min_samples_leaf,
        max_depth,
        regularization,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = LOSSES_DICT[loss](regularization)
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.rgen = check_random_state(random_state)

    def _get_max_features(self):
        if self.max_features is None:
            return self.n_features
        elif isinstance(self.max_features, numbers.Integral):
            if not (1 <= self.max_features <= self.n_features):
                raise ValueError(
                    "max_features must be in [1, num_features],"
                    f" got {self.max_features}"
                )
            return self.max_features
        elif isinstance(self.max_features, numbers.Real):
            if not (0.0 < self.max_features <= 1.0):
                raise ValueError(
                    "max_features must be in (0, 1],"
                    f" got {self.max_features}"
                )
            return round(self.max_features * self.n_features)
        elif self.max_features == "sqrt":
            return round(np.sqrt(self.n_features))
        elif self.max_features == "log2":
            return round(np.log2(self.n_features))
        else:
            raise ValueError(
                "`max_features` can be int, float or either \"sqrt\" or \"log2\","
                f" got {self.max_features}."
            )

    def _fit(self, X, y):
        y_pred = np.zeros_like(y, dtype=np.float64)
        self.estimators_ = []
        max_features = self._get_max_features()

        for _ in range(self.n_estimators):
            residuals = self.loss.grad(y, y_pred)
            tree = GBTree(
                loss=self.loss,
                max_features=max_features,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=self.rgen
            )
            tree.fit(X, residuals, y, y_pred)
            y_pred += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)

class GradientBoostingClassifier(BaseGradientBoosting):
    # TODO: implement softmax multiclass XGB
    """
    XGBoost classifier for two class problems

    Parameters
    ----------
    n_estimators : int (default: 100)
        number of trees in ensemble
    learning_rate : float (default: 0.1)
        learning rate hyperparameter
    loss : ["log_loss"]
        loss for binary classification
    max_features : int (default: None)
        number of features on each recursion step
    min_samples_leaf : int (default: 1)
        minimum possible samples in each leaf
    max_depth : int (default: inf)
        maximum depth of tree
    regularization : float (default: 1.0)
        penalty for leaf values l2 norms
    """
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        loss="log_loss",
        max_features=None,
        min_samples_leaf=1,
        max_depth=3,
        regularization=1.0,
        random_state=None
    ):
        super().__init__(
            n_estimators,
            learning_rate,
            loss,
            max_features,
            min_samples_leaf,
            max_depth,
            regularization,
            random_state
        )

    def _fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size
        super()._fit(X, y)

    def _predict_proba(self, X):
        y_pred = np.zeros(X.shape[0], np.float64)
        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)
        probas = self.loss._get_proba(y_pred)
        return probas

    def _predict(self, X):
        probas = self._predict_proba(X)
        return self.classes_.take(np.argmax(probas, axis=1), axis=0)

class GradientBoostingRegressor(BaseGradientBoosting):
    """
    XGBoost regressor

    Parameters
    ----------
    n_estimators : int (default: 100)
        number of trees in ensemble
    learning_rate : float (default: 0.1)
        learning rate hyperparameter
    loss : ["squared_error"]
        loss for regression
    max_features : int (default: None)
        number of features on each recursion step
    min_samples_leaf : int (default: 1)
        minimum possible samples in each leaf
    max_depth : int (default: inf)
        maximum depth of tree
    regularization : float (default: 1.0)
        penalty for leaf values l2 norms
    """
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        loss="squared_error",
        max_features=None,
        min_samples_leaf=1,
        max_depth=3,
        regularization=1.0,
        random_state=None
    ):
        super().__init__(
            n_estimators,
            learning_rate,
            loss,
            max_features,
            min_samples_leaf,
            max_depth,
            regularization,
            random_state
        )
    
    def _predict(self, X):
        y_pred = np.zeros(X.shape[0], dtype=np.float64)
        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred