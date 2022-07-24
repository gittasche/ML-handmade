import numpy as np

from mlhandmade.base import BaseEstimator
from .criterion import ClassificationCriterion, RegressionCriterion
from .splitter import get_best_split, split_dataset

class BaseDecisionTree(BaseEstimator):
    """
    Implementation of Decision tree
    with greedy algorithm

    Parameters
    ----------
    is_classifier : bool
        True if it is a classification tree
    criterion : BaseCriterion object
        callable criterion obeject
    max_feauters : int (default: None)
        number of features on each recursion step
    min_samples_leaf : int (default: 1)
        minimum possible samples in each leaf
    max_depth : int (default: inf)
        maximum depth of tree
    n_features : int (default: None)
        number of feauters, need for recursion
    n_classes : int (default: None)
        number of classes, need for recursion
    """
    def __init__(
        self,
        is_classifier,
        criterion,
        max_features=None,
        min_samples_leaf=1,
        max_depth=np.inf,
        random_state=0,
        n_features=None,
        n_classes=None
    ):
        self.is_classifier = is_classifier
        self.criterion = criterion
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.rgen = self._get_rgen(random_state)

        self.n_features = n_features
        self.n_classes = n_classes
        self.is_leaf = False

    @staticmethod
    def _get_rgen(random_state):
        if not isinstance(random_state, np.random.RandomState):
            return np.random.RandomState(random_state)
        return random_state

    def _fit(self, X, y):
        if self.is_classifier:
            self.n_classes = np.unique(y).shape[0]
        self._build_tree(X, y)

    def _build_tree(self, X, y):
        try:
            assert X.shape[0] > self.min_samples_leaf
            assert self.max_depth > 0

            if self.max_features is None:
                self.max_features = self.n_features
            
            features = self.rgen.choice(self.n_features, self.max_features)
            gain, feature, value = get_best_split(self.criterion, X, y, features)
            assert gain != np.inf
            
            self.feature = feature
            self.value = value

            X_left, X_right, y_left, y_right = split_dataset(X, y, feature, value)

            self.left = BaseDecisionTree(
                self.is_classifier,
                self.criterion,
                self.max_features,
                self.min_samples_leaf,
                self.max_depth - 1,
                self.rgen,
                self.n_features,
                self.n_classes
            )
            self.left._build_tree(X_left, y_left)

            self.right = BaseDecisionTree(
                self.is_classifier,
                self.criterion,
                self.max_features,
                self.min_samples_leaf,
                self.max_depth - 1,
                self.rgen,
                self.n_features,
                self.n_classes
            )
            self.right._build_tree(X_right, y_right)
        except AssertionError:
            self.is_leaf = True
            self.calculate_leaf_value(y)

    def calculate_leaf_value(self, y):
        if self.is_classifier:
            self.leaf_value = np.argmax(
                np.bincount(y, minlength=self.n_classes)
            )
        else:
            self.leaf_value = np.mean(y)

    def predict_sample(self, x):
        if not self.is_leaf:
            if x[self.feature] < self.value:
                return self.left.predict_sample(x)
            else:
                return self.right.predict_sample(x)
        else:
            return self.leaf_value

    def _predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            y_pred[i] = self.predict_sample(x)
        return y_pred

class DecisionTreeClassifier(BaseDecisionTree):
    """
    Decision tree classifier class.

    Parameters
    ----------
    criterion : ["gini", "entropy"] (default: "gini")
        impurity criterion
    max_features : int (default: None)
        number of features on each recursion step
    min_samples_leaf : int (default: 1)
        minimum possible samples in each leaf
    max_depth : int (default: inf)
        maximum depth of tree
    """
    def __init__(
        self,
        *,
        criterion="gini",
        max_features=None,
        min_samples_leaf=10,
        max_depth=10,
        random_state=0
    ):
        is_classifier = True
        criterion = ClassificationCriterion(criterion)
        super().__init__(
            is_classifier,
            criterion,
            max_features,
            min_samples_leaf,
            max_depth,
            random_state
        )

class DecisionTreeRegressor(BaseDecisionTree):
    """
    Decision tree regressor class.

    Parameters
    ----------
    criterion : ["mse", "mae"] (default: "mse")
        impurity criterion
    max_features : int (default: None)
        number of features on each recursion step
    min_samples_leaf : int (default: 1)
        minimum possible samples in each leaf
    max_depth : int (default: inf)
        maximum depth of tree
    """
    def __init__(
        self,
        *,
        criterion="mse",
        max_features=None,
        min_samples_leaf=10,
        max_depth=10,
        random_state=0
    ):
        is_classifier = False
        criterion = RegressionCriterion(criterion)
        super().__init__(
            is_classifier,
            criterion,
            max_features,
            min_samples_leaf,
            max_depth,
            random_state
        )