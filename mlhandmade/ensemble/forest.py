from abc import abstractmethod
import numpy as np

from ..base import BaseEstimator
from ..tree.tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils.validations import check_random_state

def _get_n_sampes_bootstrap(n_samples, max_samples):
    if max_samples is None:
        return n_samples
    elif isinstance(max_samples, int):
        if not (1 <= max_samples <= n_samples):
            raise ValueError(
                f"max_samples must be in range 1 to {n_samples}, got {max_samples}"
            )
        return max_samples
    elif isinstance(max_samples, float):
        if not (0 < max_samples <= 1):
            raise ValueError(
                f"max_samples must be in range (0.0, 1.0], got {max_samples}"
            )
        return round(n_samples * max_samples)

class BaseForest(BaseEstimator):
    """
    Base class for random forest estimators.

    Parameters
    ----------
    base_tree : BaseDecisionTree class
        DecisionTreeClassifier in case of classification
        DecisionTreeRegressor in case of regression
    estimator_kwargs : dict
        keyword arguments for estimator
    n_estimators : int (default: 10)
        number of estimators
    bootstrap : bool (default: False)
        if True train estimators on random samples with replacement
        else train all estimators on whole data
    max_samples : int, float or None (default: None)
        - int: number of samples in bootstrap
        - float: fraction of whole data in bootstrap
        - None: number of samples in bootstrap = number of samples in data
    """
    def __init__(
        self,
        base_tree,
        estimator_kwargs,
        n_estimators=10,
        bootstrap=False,
        max_samples=None,
        random_state=0
    ):
        self.base_tree = base_tree
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.rgen = check_random_state(random_state)
        self.estimator_kwargs = estimator_kwargs

    def _fit(self, X, y):
        # get different random states for each tree in ensemble
        seeds = self.rgen.randint(1000, size=self.n_estimators)
        trees = [
            self.base_tree(**self.estimator_kwargs, random_state=seed)
            for seed in seeds
        ]

        self.estimators_ = []

        n_samples_bootstrap = _get_n_sampes_bootstrap(self.n_samples, self.max_samples)
        for tree in trees:
            if self.bootstrap:
                # get bootstrap data from different random states
                bootstrap_idx = tree.get_rgen.choice(
                    np.arange(self.n_samples),
                    size=n_samples_bootstrap,
                    replace=True,
                )
                tree.fit(X[bootstrap_idx], y[bootstrap_idx])
            else:
                tree.fit(X, y)

        self.estimators_.extend(trees)

    @abstractmethod
    def _predict(self, X):
        raise NotImplementedError()

class RandomForestClassifier(BaseForest):
    """
    Implemntation of random forest classifier

    Parameters
    ----------
    n_estimators : int (default: 10)
        number of trees
    criterion : ["gini", "entropy"] (default: "gini")
        criterion for DecisionTreeClassifier
    max_depth : int or inf (default: inf)
        max depth of trees
    min_samples_leaf : int (default: 1)
        min samples in leaf to split node
    max_features : "sqrt", int or None (default: "sqrt")
        features to consider on each split
    bootstrap : bool (default: True)
        use bootstrap sample selection
    max_samples : int, float or None (default: None)
        - int: number of samples in bootstrap
        - float: fraction of whole data in bootstrap
        - None: number of samples in bootstrap = number of samples in data
    """
    def __init__(
        self,
        n_estimators=10,
        *,
        criterion="gini",
        max_depth=np.inf,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        max_samples=None,
        random_state=0
    ):
        estimator_kwargs={
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }
        super().__init__(
            base_tree=DecisionTreeClassifier,
            estimator_kwargs=estimator_kwargs,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            max_samples=max_samples,
            random_state=random_state
        )

    def _fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        super()._fit(X, y)
    
    def _predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def predict_proba(self, X):
        all_proba = np.mean(
            [e.predict_proba(X) for e in self.estimators_],
            axis=0
        )
        return all_proba

class RandomForestRegressor(BaseForest):
    """
    Implemntation of random forest classifier

    Parameters
    ----------
    n_estimators : int (default: 10)
        number of trees
    criterion : ["mse", "mae"] (default: "mse")
        criterion for DecisionTreeClassifier
    max_depth : int or inf (default: inf)
        max depth of trees
    min_samples_leaf : int (default: 1)
        min samples in leaf to split node
    max_features : "sqrt", int or None (default: None)
        features to consider on each split
    bootstrap : bool (default: True)
        use bootstrap sample selection
    max_samples : int, float or None (default: None)
        - int: number of samples in bootstrap
        - float: fraction of whole data in bootstrap
        - None: number of samples in bootstrap = number of samples in data
    """
    def __init__(
        self,
        n_estimators=10,
        *,
        criterion="mse",
        max_depth=np.inf,
        min_samples_leaf=1,
        max_features=None,
        bootstrap=True,
        max_samples=None,
        random_state=0
    ):
        estimator_kwargs={
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }
        super().__init__(
            base_tree=DecisionTreeRegressor,
            estimator_kwargs=estimator_kwargs,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            max_samples=max_samples,
            random_state=random_state
        )

    def _fit(self, X, y):
        return super()._fit(X, y)

    def _predict(self, X):
        y_hat = np.mean(
            [e.predict(X) for e in self.estimators_],
            axis=0
        )
        return y_hat