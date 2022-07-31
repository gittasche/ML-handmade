from abc import abstractmethod
import numpy as np
import numbers

from ..base import BaseEstimator
from .criterion import ClassificationCriterion, RegressionCriterion
from .splitter import get_best_split, get_split_mask
from ..utils.validations import check_random_state, check_sample_weight

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
    max_feauters : None, int, float or in ["sqrt", "log2"] (default: None)
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
        self.rgen = check_random_state(random_state)

        self.n_features = n_features
        self.n_classes = n_classes
        self.is_leaf = False

    def _fit(self, X, y, sample_weight=None):
        """
        Fit decision tree.

        Parameters
        ----------
        X : np.ndarray of shape (N, D)
            array of data
        y : np.ndarray of shape (N,)
            array of targets
        sample_weight : array-like of shape (N,), number, None
            array of weights for all statistics: average, probability, median
            sample_weight is needed for some ensemble algorithms such as AdaBoost
        """
        if self.is_classifier:
            self.classes_ = np.unique(y)
            self.n_classes = self.classes_.shape[0]
        
        self.max_features = self._get_max_features()
        sample_weight = check_sample_weight(sample_weight, X)
        self._build_tree(X, y, sample_weight)

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

    def _build_tree(self, X, y, sample_weight):
        try:
            assert self._check_leaf(X.shape[0])
            
            features = self.rgen.choice(self.n_features, self.max_features)
            gain, feature, value = get_best_split(self.criterion, X, y, features, sample_weight)
            assert gain != np.inf
            
            self.feature = feature
            self.value = value

            left_mask, right_mask = get_split_mask(X, feature, value)

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
            self.left._build_tree(X[left_mask], y[left_mask], sample_weight[left_mask])

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
            self.right._build_tree(X[right_mask], y[right_mask], sample_weight[right_mask])
        except AssertionError:
            self.is_leaf = True
            self.calculate_leaf_value(y, sample_weight)

    def _check_leaf(self, num_samples_node):
        samples_condition = num_samples_node > self.min_samples_leaf
        depth_condition = self.max_depth > 0
        return samples_condition & depth_condition

    def calculate_leaf_value(self, y, sample_weight):
        if self.is_classifier:
            self.leaf_value = np.bincount(y, weights=sample_weight, minlength=self.n_classes) / np.sum(sample_weight)
        else:
            self.leaf_value = np.average(y, weights=sample_weight)

    def predict_sample(self, x):
        if not self.is_leaf:
            if x[self.feature] < self.value:
                return self.left.predict_sample(x)
            else:
                return self.right.predict_sample(x)
        else:
            return self.leaf_value

    @abstractmethod
    def _predict(self, X):
        raise NotImplementedError()

    @property
    def get_rgen(self):
        return self.rgen

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
        min_samples_leaf=1,
        max_depth=np.inf,
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

    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], self.n_classes))
        for i, x in enumerate(X):
            probas[i] = self.predict_sample(x)
        return probas

    def _predict(self, X):
        return self.classes_.take(np.argmax(self.predict_proba(X), axis=-1), axis=0)

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
        min_samples_leaf=1,
        max_depth=np.inf,
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

    def _predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            y_pred[i] = self.predict_sample(x)
        return y_pred