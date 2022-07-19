import numpy as np

from mlhandmade.base.base import BaseEstimator
from mlhandmade.preprocessing.data_preprocessiong import add_bias_feature
from .optimizers import *

optimizer_dict = {"gd" : GD, "sgd" : SGD, "batch_gd" : BatchGD, "sag" : SAG}
exact_methods = ["direct", "svd", "qr"]

class LinearRegressor(BaseEstimator):
    """
    Logistic Regression class implemented with different
    methods: exact(matrix decompositions) and approximate(gradient).
    Exact methods faster due to numpy implementation

    Objective loss:
    L(X, y, w) = ||Xw - y||^2_2

    Attributes
    ----------
    method : str
        Optimization algorithm to minimize loss
        Availible optimizers:
        ["direct", "svd", "qr", "gd", "sgd", "batch_gd", "sag"]
    epochs : int
        number of optimization iterations
    tol : float
        numerical tolerance
    random_state : int
        random state to debug calculations
    optimizer_settings : kwargs
        kwargs for chosen optimizer
    """
    def __init__(
        self,
        method: str,
        epochs: int = None,
        tol: float = 1e-3,
        random_state: int = 1,
        **optimizer_settings
    ) -> None:
        self._validate_input(method, epochs)
        if method in optimizer_dict:
            self.optimizer = optimizer_dict[method](**optimizer_settings)
        self.method = method
        
        self.epochs = epochs
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _validate_input(method, epochs):
        if method in optimizer_dict:
            if epochs is None:
                raise ValueError("epochs must be positive int if method is gradient")
        elif method in exact_methods:
            pass
        else:
            raise ValueError(f"Unknown method, got {method}.\nAvailible methods:\
            \nexact - {exact_methods}\ngradient - {list(optimizer_dict.keys())}")

    @staticmethod
    def _loss_grad(X, y, w):
        if y.ndim == 0:
            return 2 * X * (X @ w - y)
        else:
            return 2 * X.T @ (X @ w - y)

    def _fit(self, X, y):
        X = add_bias_feature(X)
        
        if self.method in optimizer_dict:
            rgen = np.random.RandomState(self.random_state)
            self.w_ = rgen.normal(loc=0.0, scale=0.01, size=self.n_features + 1)
            for _ in range(self.epochs):
                self.w_ = self.optimizer.update(self._loss_grad, X, y, self.w_)
                if np.linalg.norm(X @ self.w_ - y) < self.tol:
                    break
        
        elif self.method == "direct":
            self.w_ = np.linalg.inv(X.T @ X) @ X.T @ y

        elif self.method == "svd":
            # w = V x S^-1 x U
            self.w_ = np.linalg.pinv(X) @ y
        
        elif self.method == "qr":
            Q, R = np.linalg.qr(X)
            self.w_ = np.linalg.inv(R) @ Q.T @ y

    def _predict(self, X):
        return X @ self.w_[1:] + self.w_[0]

class RidgeRegressor(BaseEstimator):
    """
    Regularized linear regression implemented with svd

    Objective loss:
    L(X, y, w) = ||Xw - y||^2_2 + alpha * ||w||^2_2

    Attributes
    ----------
    alpha : float
        penalty parameter
    """
    def __init__(
        self,
        alpha: float = 0,
    ) -> None:
        self.alpha = alpha

    @staticmethod
    def _solve_svd(X, y, alpha):
        # w = V x (S^T x S + alpha * I)^-1 x S^T x U^T x y
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        idx = s > 1e-15
        s_nnz = s[idx]
        UTy = U.T @ y
        d = np.zeros(s.size, dtype=X.dtype)
        d[idx] = s_nnz / (s_nnz**2 + alpha)
        d_UT_y = d * UTy
        return Vt.T @ d_UT_y

    def _fit(self, X, y):
        X = add_bias_feature(X)
        self.w_ = self._solve_svd(X, y, self.alpha)
        
    def _predict(self, X):
        return X @ self.w_[1:] + self.w_[0]