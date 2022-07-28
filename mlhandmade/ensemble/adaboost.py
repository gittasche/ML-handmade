from abc import abstractmethod
import numpy as np
from scipy.special import xlogy

from ..base import BaseEstimator
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils.validations import check_random_state, _num_samples

class BaseAdaBoost(BaseEstimator):
    """
    Base class for AdaBoost algorithms
    base on decision trees

    Parameters
    ----------
    base_tree : BaseDecisionTree class
        DecisionTreeClassifier in case of classification
        DecisionTreeRegressor in case of regression
    estimator_kwargs : dict
        keyword arguments for estimator
    n_estimators : int (default: 50)
        number of estimators
    learning_rate : float (default: 1.0)
        hyperparameter as a `n_estimators`. `learning_rate`
        controls contribution of each estimator in ensemble
    """
    def __init__(
        self,
        base_tree,
        estimator_kwargs,
        n_estimators=50,
        learning_rate=1.0,
        random_state=0
    ):
        self.base_tree = base_tree
        self.n_estimators = n_estimators
        self.estimator_kwargs = estimator_kwargs
        self.learning_rate = learning_rate
        self.rgen = check_random_state(random_state)

    def _fit(self, X, y):
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        sample_weight = np.ones(self.n_samples) / self.n_samples

        for iboost in range(self.n_estimators):
            # get different random states for each estimator to avoid
            # choice of the same feature due to `max_feature`
            random_seed = self.rgen.randint(1000)
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost, X, y, sample_weight, random_state=check_random_state(random_seed)
            )

            if sample_weight is None: # case of bad fitting
                break
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            if estimator_error == 0: # case of perfect fit
                break

            sample_weight_sum = np.sum(sample_weight)

            if sample_weight_sum <= 0: # sum sample_weight in tree should be positive
                break

            if iboost < self.n_estimators - 1:
                sample_weight /= sample_weight_sum

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, random_state):
        raise NotImplementedError()

def _samme_proba(estimator, n_classes, X):
    """
    Get class probabilities in case of "SAMME.R"
    """
    proba = estimator.predict_proba(X)
    np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
    log_proba = np.log(proba)

    return (n_classes - 1) * (
        log_proba - (1.0 / n_classes) * log_proba.sum(axis=1)[:, np.newaxis]
    )

class AdaBoostClassifier(BaseAdaBoost):
    """
    AdaBoost multiclass classifier with SAMME algorithm

    Parameters
    ----------
    n_estimators : int (default: 10)
        number of trees
    criterion : ["gini", "entropy"] (default: "gini")
        criterion for DecisionTreeClassifier
    max_depth : int or inf (default: 1)
        max depth of trees, default value is 1, i.e.
        decision stumps
    min_samples_leaf : int (default: 1)
        min samples in leaf to split node
    max_features : "sqrt", int or None (default: "sqrt")
        features to consider on each split
    learning_rate : float (default: 1.0)
        hyperparameter as a `n_estimators`. `learning_rate`
        controls contribution of each estimator in ensemble
    algorithm : ["SAMME", "SAMME.R"] (default: SAMME.R)
        "SAMME" algorithm uses discrete class labels,
        "SAMME.R" uses their probabilities instead,
        so "SAMME.R" is more efficient
    """
    def __init__(
        self,
        n_estimators=50,
        *,
        criterion="gini",
        max_depth=1,
        min_samples_leaf=1,
        max_features="sqrt",
        learning_rate=1.0,
        algorithm="SAMME.R",
        random_state=0
    ):
        if algorithm not in ("SAMME", "SAMME.R"):
            raise ValueError(
                "Algorithm must be \"SAMME\" or \"SAMME.R\","
                f" got {algorithm!r} instead."
            )
        
        self.algorithm = algorithm

        estimator_kwargs={
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features
        }
        super().__init__(
            base_tree=DecisionTreeClassifier,
            estimator_kwargs=estimator_kwargs,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )

    def _fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        return super()._fit(X, y)

    def _boost(self, iboost, X, y, sample_weight, random_state):
        self.estimator_kwargs["random_state"] = random_state
        if self.algorithm == "SAMME.R":
            return self._boost_real(iboost, X, y, sample_weight)
        elif self.algorithm == "SAMME":
            return self._boost_discrete(iboost, X, y, sample_weight)
    
    def _boost_real(self, iboost, X, y, sample_weight):
        tree = self.base_tree(**self.estimator_kwargs)
        tree.fit(X, y, sample_weight=sample_weight)
        self.estimators_.append(tree)

        y_predict_proba = tree.predict_proba(X)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1), axis=0)

        incorrect = y_predict != y

        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        classes = self.classes_
        n_classes = self.n_classes_
        y_codes = np.array([-1.0 / (n_classes - 1), 1.0])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        proba = y_predict_proba
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        estimator_weight = (
            -1.0
            * self.learning_rate
            * ((n_classes - 1.0) / n_classes)
            * xlogy(y_coding, y_predict_proba).sum(axis=1)
        )

        if iboost != self.n_estimators - 1:
            sample_weight *= np.exp(
                estimator_weight * ((sample_weight > 0) | (estimator_weight < 0))
            )
        
        return sample_weight, 1.0, estimator_error

    def _boost_discrete(self, iboost, X, y, sample_weight):
        tree = self.base_tree(**self.estimator_kwargs)
        tree.fit(X, y, sample_weight=sample_weight)
        self.estimators_.append(tree)

        y_predict = tree.predict(X)

        incorrect = y_predict != y

        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        n_classes = self.n_classes_

        if estimator_error >= 1.0 - (1.0 / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError(
                    "AdaBoostClassifier is worse than random,"
                    " ensemble can not be fit."
                )
            return None, None, None

        estimator_weight = self.learning_rate * (
            np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0)
        )

        if iboost != self.n_estimators - 1:
            sample_weight = np.exp(
                np.log(sample_weight)
                + estimator_weight * incorrect * (sample_weight > 0)
            )

        return sample_weight, estimator_weight, estimator_error

    def decision_function(self, X):
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]

        if self.algorithm == "SAMME.R":
            pred = sum(
                _samme_proba(estimator, n_classes, X) for estimator in self.estimators_
            )
        elif self.algorithm == "SAMME":
            pred = sum(
                (estimator.predict(X) == classes).T * w
                for estimator, w in zip(self.estimators_, self.estimator_weights_)
            )

        pred /= self.estimator_weights_.sum()
        return pred

    def _predict(self, X):
        pred = self.decision_function(X)
        
        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

class AdaBoostRegressor(BaseAdaBoost):
    """
    AdaBoost regressor with AdaBoost.R2 algorithm

    Parameters
    ----------
    n_estimators : int (default: 10)
        number of trees
    criterion : ["mse", "mae"] (default: "mse")
        criterion for DecisionTreeClassifier
    max_depth : int or inf (default: 3)
        max depth of trees
    min_samples_leaf : int (default: 1)
        min samples in leaf to split node
    max_features : "sqrt", int or None (default: None)
        features to consider on each split
    loss : ["linear", "square", "exponential"] (default: "linear")
        loss function to use when updating the weights after each
        boosting iteration.
    """
    def __init__(
        self,
        n_estimators=50,
        *,
        criterion="mse",
        max_depth=3,
        min_samples_leaf=1,
        max_features=None,
        learning_rate=1.0,
        loss="linear",
        random_state=0,
    ):
        if loss not in ("linear", "square", "exponential"):
            raise ValueError(
                    "Loss must be \"linear\", \"square\" or \"exponential\","
                    f" got {loss!r} instead."
                )
        
        self.loss = loss
        estimator_kwargs={
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features
        }
        super().__init__(
            base_tree=DecisionTreeRegressor,
            estimator_kwargs=estimator_kwargs,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )

    def _fit(self, X, y):
        return super()._fit(X, y)

    def _boost(self, iboost, X, y, sample_weight, random_state):
        self.estimator_kwargs["random_state"] = random_state
        tree = self.base_tree(**self.estimator_kwargs)
        bootstrap_idx = random_state.choice(
            np.arange(self.n_samples),
            size=self.n_samples,
            replace=True,
            p=sample_weight,
        )
        tree.fit(X[bootstrap_idx], y[bootstrap_idx])
        self.estimators_.append(tree)
        y_predict = tree.predict(X)

        error_vect = np.abs(y_predict - y)
        sample_mask = sample_weight > 0
        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vect = error_vect[sample_mask]

        error_max = masked_error_vect.max()
        if error_max != 0:
            masked_error_vect /= error_max

        if self.loss == "square":
            masked_error_vect **= 2
        elif self.loss == "exponential":
            masked_error_vect = 1.0 - np.exp(-masked_error_vect)

        estimator_error = (masked_sample_weight * masked_error_vect).sum()

        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0
        elif estimator_error >= 0.5:
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None
        
        beta = estimator_error / (1.0 - estimator_error)
        estimator_weight = self.learning_rate * np.log(1.0 / beta)

        if iboost != self.n_estimators - 1:
            sample_weight[sample_mask] *= np.power(
                beta, (1.0 - masked_error_vect) * self.learning_rate
            )
        
        return sample_weight, estimator_weight, estimator_error

    def _get_median_predict(self, X, limit):
        prediction = np.array([e.predict(X) for e in self.estimators_[:limit]]).T
        sorted_idx = np.argsort(prediction, axis=1)
        weight_cdf = np.cumsum(self.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(_num_samples(X)), median_idx]

        return prediction[np.arange(_num_samples(X)), median_estimators]

    def _predict(self, X):
        return self._get_median_predict(X, len(self.estimators_))