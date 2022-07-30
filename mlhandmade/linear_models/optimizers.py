from abc import abstractmethod
import numpy as np
from ..preprocessing import data_shuffle
from ..utils.validations import check_random_state

class BaseOptimizer:
    """
    Base class for optimizers of finite sums.
    """
    @abstractmethod
    def update(self, grad, X, y, w):
        raise NotImplementedError()

class GD(BaseOptimizer):
    """
    Gradient descent optimizer:

    w_new = w - eta * mean(grad(L(X, y, w))),
    L - loss
    X - full dataset samples
    y - full dataset targets
    w - weights

    Parameters
    ----------
    eta : float
        learning rate
        
    Returns
    -------
    w : ndarray of shape(D+1,)
        weights
    """
    def __init__(self, eta):
        self.eta = eta

    def update(self, grad, X, y, w):
        w -= self.eta * np.mean(grad(X, y, w), axis=0)
        return w

class SGD(BaseOptimizer):
    """
    Stochastic gradient descent optimizer:

    w_new = w - grad(L(xi, yi, w))
    L - loss
    xi - one sample from dataset
    yi - one target from dataset
    w - weights

    Parameters
    ----------
    eta : float
        learning rate
    random_state : int, None, np.random.RandomState (default: 0)
        random seed to reproduce calculations
    shuffle : bool (default: True)
        shuffle data before each iteration if True
        
    Returns
    -------
    w : ndarray of shape(D+1,)
        weights
    """
    def __init__(self, eta, random_state=0, shuffle=True):
        self.eta = eta
        self.rgen = check_random_state(random_state)
        self.shuffle = shuffle

    def update(self, grad, X, y, w):
        if self.shuffle:
            X, y = data_shuffle(X, y, random_state=self.rgen)
        for xi, target in zip(X, y):
            w -= self.eta * grad(xi, target, w)
        return w

class BatchGD(BaseOptimizer):
    """
    Batch gradient descent optimizer

    w_new = w - mean(grad(L(Xs, ys, w))),
    L - loss
    Xs - random subsample from X (batch)
    ys - random subsample from y (batch)
    w - weights

    Parameters
    ----------
    eta : float
        learning rate
    batch_size : int in (0, N]
        size of a batch
    random_state : int, None, np.random.RandomState (default: 0)
        random seed to reproduce calculations
    shuffle : bool (default: True)
        shuffle data before each iteration if True

    Returns
    -------
    w : ndarray of shape(D+1,)
        weights

    SGD is a special case of BatchGD with `batch_size=1`
    """
    def __init__(self, eta, batch_size, random_state=0, shuffle=True):
        self.eta = eta
        self.batch_size = batch_size
        self.rgen = check_random_state(random_state)
        self.shuffle = shuffle

    def update(self, grad, X, y, w):
        for xb, tb in self._batch_iterator(X, y):
            w -= self.eta * np.mean(grad(xb, tb, w), axis=0)
        return w

    def _batch_iterator(self, X, y):
        if self.batch_size > X.shape[0] or self.batch_size <= 0:
            raise ValueError(
                    "batch_size must be in (0, num_sampes],"
                    f" got {self.batch_size}."
                )
        
        idx = np.arange(X.shape[0])
        if self.shuffle:
            self.rgen.shuffle(idx)

        for i in range(0, X.shape[0], self.batch_size):
            begin, end = i, min(i + self.batch_size, X.shape[0])
            yield X[idx[begin:end]], y[idx[begin:end]]

class SAG(BaseOptimizer):
    """
    Stochastic average gradient optimizer.
    This algorithm calculates gradients only on first
    iteration, then it updates only one gradient component
    on each iteration. In other its similar to GD

    Parameters
    ----------
    eta : float
        learning rate
    random_state : int, None, np.random.RandomState (default: 0)
        random seed to reproduce calculations
        
    Returns
    -------
    w : ndarray of shape(D+1,)
        weights
    """
    def __init__(self, eta, random_state=0):
        self.eta = eta
        self.rgen = check_random_state(random_state)

    @staticmethod
    def _setup_grads(grad, X, y, w):
        grads = grad(X, y, w)
        return grads

    def update(self, grad, X, y, w):
        if not hasattr(self, "grads_"):
            self.grads_ = self._setup_grads(grad, X, y, w)
        k = self.rgen.randint(0, X.shape[0])
        self.grads_[k] = grad(X[k], y[k], w)
        w -= self.eta * np.mean(self.grads_, axis=0)
        return w