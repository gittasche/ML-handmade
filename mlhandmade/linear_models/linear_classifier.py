import numpy as np

from mlhandmade.base.base import BaseEstimator
from .losses import *
from .optimizers import *
from mlhandmade.preprocessing.data_preprocessiong import add_bias_feature, onehot

loss_dict = {"perceptron" : PerceptronLoss, "logistic" : LogisticLoss,
             "hinge" : HingeLoss, "sigmoid" : SigmoidLoss}

optimizer_dict = {"gd" : GD, "sgd" : SGD, "batch_gd" : BatchGD, "sag" : SAG}

class LinearClassifier(BaseEstimator):
    """
    Linear classifier class

    Attributes
    ----------
    loss : str
        Loss function to minimize.
        Availible losses:
        ["perceptron", "logistic", "hinge", "sigmoid"]
    optimizer : str
        Optimization algorithm to minimize loss
        Availible optimizers:
        ["gd", "sgd", "batch_gd", "sag"]
    epochs : int
        number of optimization iterations
    tol : float
        numerical tolerance
    random_state : int
        random state to debug calculations
    optimizer_settings : kwargs
        kwargs for chosen optimizer
    """
    def __init__(self, loss, optimizer, epochs, tol=1e-3, random_state=1, **optimizer_settings):
        self.loss = loss_dict[loss]()
        self.optimizer = optimizer_dict[optimizer](**optimizer_settings)
        self.epochs = epochs
        self.tol = tol
        self.random_state = random_state

    def _fit(self, X: np.ndarray, y: np.ndarray):
        X = add_bias_feature(X)
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=self.n_features + 1)
        for _ in range(self.epochs):
            self.w_ = self.optimizer.update(self.loss.grad, X, y, self.w_)
            if np.mean(self.loss._loss(X, y, self.w_)) < self.tol:
                break

    def decision_function(self, X: np.ndarray):
        return X @ self.w_[1:] + self.w_[0]

    def _predict(self, X):
        return np.where(self.decision_function(X) > 0, 1, -1)

class SoftmaxRegressor(BaseEstimator):
    """
    Softmax regression class (multinominal logistic regression).
    Softmax function can't be represented as finite sum,
    so we can't use any stochastic descents here.

    Attributes
    ----------
    eta : float
        learning rate
    epochs : int
        number of optimization iterations
    random_state : int
        random state to debug calculations
    """
    def __init__(self, eta, epochs, random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

    @staticmethod
    def _softmax(X, w):
        exp_a = np.exp(X @ w)
        return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

    def _fit(self, X, y):
        X = add_bias_feature(X)
        if y.ndim == 1:
            y = onehot(y)
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=(self.n_features + 1, y.shape[1]))
        for _ in range(self.epochs):
            grad = X.T @ (self._softmax(X, self.w_) - y)
            self.w_ -= self.eta * grad

    def _predict(self, X: np.ndarray):
        X = add_bias_feature(X)
        return np.argmax(self._softmax(X, self.w_), axis=-1)