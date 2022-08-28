import numpy as np
from abc import abstractmethod
from scipy.special import expit, logsumexp

class BaseLoss:
    def __init__(self, regularization=1.0):
        self.regularization = regularization

    @abstractmethod
    def __call__(self, y_true, y_pred):
        raise NotImplementedError()

    @abstractmethod
    def grad(self, y_true, y_pred, **kargs):
        raise NotImplementedError()

    @abstractmethod
    def hess(self, y_true, y_pred):
        raise NotImplementedError()

    def gain(self, residuals, y_true, y_pred):
        # use equality (6) from xgboost arxiv paper
        num = residuals.sum() ** 2
        denom = self.hess(y_true, y_pred).sum() + self.regularization
        return 0.5 * num / denom

class LeastSquaresLoss(BaseLoss):
    def __call__(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def grad(self, y_true, y_pred, **kargs):
        return y_true - y_pred

    def hess(self, y_true, y_pred):
        return np.ones_like(y_true, dtype=np.float64)

class LogisticLoss(BaseLoss):
    def __call__(self, y_true, y_pred):
        return -2 * np.mean(
            y_true * y_pred - np.logaddexp(0, y_pred)
        )

    def grad(self, y_true, y_pred, **kargs):
        return y_true - expit(y_pred)

    def hess(self, y_true, y_pred):
        exp = expit(y_pred)
        return exp * (1 - exp)
    
    def _get_proba(self, y_pred):
        probas = np.ones((y_pred.size, 2), dtype=np.float64)
        probas[:, 1] = expit(y_pred)
        probas[:, 0] -= probas[:, 1]
        return probas
