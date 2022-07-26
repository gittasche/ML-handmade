from abc import abstractmethod
import numpy as np
from ..preprocessing import data_shuffle

class BaseOptimizer:
    """
    Base class for optimizers of finite sums.
    """
    @abstractmethod
    def update(self, grad, X, y, w):
        raise NotImplementedError()

class GD(BaseOptimizer):
    def __init__(self, eta):
        self.eta = eta

    def update(self, grad, X, y, w):
        w -= self.eta * np.mean(grad(X, y, w), axis=0)
        return w

class SGD(BaseOptimizer):
    def __init__(self, eta):
        self.eta = eta

    def update(self, grad, X, y, w):
        X, y = data_shuffle(X, y)
        for xi, target in zip(X, y):
            w -= self.eta * grad(xi, target, w)
        return w

class BatchGD(BaseOptimizer):
    def __init__(self, eta, batch_size, random_state=1):
        self.eta = eta
        self.batch_size = batch_size
        self.rgen = self._setup_rgen(random_state)

    @staticmethod
    def _setup_rgen(random_state):
        return np.random.RandomState(random_state)

    def update(self, grad, X, y, w):
        for xb, tb in self._batch_iterator(X, y):
            w -= self.eta * np.mean(grad(xb, tb, w), axis=0)
        return w

    def _batch_iterator(self, X, y):
        idx = np.arange(X.shape[0])
        self.rgen.shuffle(idx)

        for i in range(0, X.shape[0], self.batch_size):
            begin, end = i, min(i + self.batch_size, X.shape[0])
            yield X[idx[begin:end]], y[idx[begin:end]]

class SAG(BaseOptimizer):
    def __init__(self, eta, random_state=1):
        self.eta = eta
        self.rgen = self._setup_rgen(random_state)

    @staticmethod
    def _setup_grads(grad, X, y, w):
        grads = grad(X, y, w)
        return grads

    @staticmethod
    def _setup_rgen(random_state):
        return np.random.RandomState(random_state)

    def update(self, grad, X, y, w):
        if not hasattr(self, "grads_"):
            self.grads_ = self._setup_grads(grad, X, y, w)
        k = self.rgen.randint(0, X.shape[0])
        self.grads_[k] = grad(X[k], y[k], w)
        w -= self.eta * np.mean(self.grads_, axis=0)
        return w