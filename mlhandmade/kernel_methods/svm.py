import numpy as np

from ..base import BaseEstimator
from .kernels import (
    LinearKernel,
    PolyKernel,
    RBFKernel
)


kernels = {
    "linear" : LinearKernel,
    "poly" : PolyKernel,
    "rbf" : RBFKernel
}

class BaseSVM(BaseEstimator):
    """
    Base class for SVM implements QP solver

    Parameters
    ----------
    C : float
        constraint for lagrange multipliers
        in dual problem
    """
    def __init__(self, C):
        self.C = C
    
    def _solve_qp(self, coef, grad, t, Q, tol):
        """
        Solve QP problem with one linear constraint [1]:

        min ->coef | 0.5 * coef^T @ Q @ coef - grad^T @ coef
        subject to | t^T @ coef = 0
                   | 0 <= coef[i] <= C, i = 1, ..., N

        Reference:
        [1] `Chang and Lin, LIBSVM: A Library for Support Vector Machines.`.
        [2] ctgk/PRML/prml/kernel on github.

        Parameters
        ----------
        X : (N, D) np.ndarray
            training independent variable
        t : (N,) np.ndarray
            training dependent variable
            binary -1 or 1
        tol : float, optional
            numerical tolerance (the default is 1e-8)
        """
        while True:
            # Select working set with WSS 1 "maximal violating pair"[1]
            # i = argmax{-y * grad | y from I_up}
            # j = argmin{-y * grad | y from I_low}, where
            # I_up = {y | (coef < C) & (y = 1) or (coef > 0) & (y = -1)}
            # I_low = {y | (coef < C) & (y = -1) or (coef > 0) & (y = 1)}
            tg = t * grad
            mask_up = (t == 1) & (coef < self.C - tol)
            mask_up |= (t == -1) & (coef > tol)
            mask_down = (t == -1) & (coef < self.C - tol)
            mask_down |= (t == 1) & (coef > tol)
            i = np.where(mask_up)[0][np.argmax(tg[mask_up])]
            j = np.where(mask_down)[0][np.argmin(tg[mask_down])]
            
            # check convergence
            if tg[i] < tg[j] + tol:
                # b must satisfy conditions tg[i] < b < tg[j],
                # so it's no matter how we sum tg's, average is a good choice[1]
                self.b = 0.5 * (tg[i] + tg[j])
                break
            # Solve two-variable problem
            # A, B - constraints: (a_i, a_j) in the box [0, C] x [0, C]
            A = self.C - coef[i] if t[i] == 1 else coef[i]
            B = coef[j] if t[j] == 1 else self.C - coef[j]
            direction = (tg[i] - tg[j]) / (Q[i, i] - 2 * Q[i, j] + Q[j, j])
            # limit coef's if direction is too large
            direction = min(A, B, direction)
            coef[i] += direction * t[i]
            coef[j] -= direction * t[j]
            grad -= direction * t * (Q[i] - Q[j])
        
        return coef

    def decision_function(self, X):
        """
        SVM classifiers can be used with multiclass strategies
        """
        raise NotImplementedError()

class SupportVectorClassifier(BaseSVM):
    """
    support vector classifier algorithm requires solution of 
    the next QP problem:

    min ->coef | 0.5 * coef^T @ Q @ coef - grad^T @ coef
    subject to | t^T @ coef = 0
               | 0 <= coef[i] <= C, i = 1, ..., N

    where Q = Gram, Gram = K(X, X)
    grad - ones(N)
    coef - zeros(N)
    t - class labels y

    Parameters
    ----------
    C : float (default: np.inf)
        inverse regularization parameter (inf - no penalty)
    kernel : str (default: "linear")
        kernel insted of inner product
    kernel_kwargs : kwargs
        parameters of kernel
    """
    def __init__(self, C=np.inf, kernel="linear", **kernel_kwargs):
        if kernel not in kernels:
            raise ValueError(f"Unkown kernel: avilible only {list(kernels.keys())}")
        self.kernel = kernels[kernel](**kernel_kwargs)
        super().__init__(C)

    def _fit(self, X, y, tol=1e-8):
        """
        estimate support vectors and their parameters
        Parameters
        ----------
        X : (N, D) np.ndarray
            training independent variable
        y : (N,) np.ndarray
            training dependent variable
            binary -1 or 1
        tol : float, optional
            numerical tolerance (the default is 1e-8)
        """
        N = len(y)
        coef = np.zeros(N) # alphas(lagrangian multipliers)
        grad = np.ones(N)
        t = y
        Q = self.kernel(X, X)
        alpha = super()._solve_qp(coef, grad, t, Q, tol)
        support_mask = alpha > tol
        self.a = alpha[support_mask]
        self.X = X[support_mask]
        self.y = y[support_mask]

    def _predict(self, X):
        """
        predict labels of the input

        Parameters
        ----------
        x : (sample_size, n_features) ndarray
            input

        Returns
        -------
        label : (sample_size,) ndarray
            predicted labels
        """
        y = self.decision_function(X)
        labels = np.sign(y)
        return labels

    def decision_function(self, X):
        """
        calculate distance from the decision boundary

        Parameters
        ----------
        x : (sample_size, n_features) ndarray
            input

        Returns
        -------
        distance : (sample_size,) ndarray
            distance from the boundary
        """
        distance = np.sum(
            self.a * self.y
            * self.kernel(X, self.X),
            axis=-1) + self.b
        return distance

class SupportVectorRegressor(BaseSVM):
    """
    support vector classifier algorithm requires solution of 
    the next QP problem:

    min ->coef | 0.5 * coef^T @ Q @ coef - grad^T @ coef
    subject to | t^T @ coef = 0
               | 0 <= coef[i] <= C, i = 1, ..., N

    where Q = [[Gram, -Gram]
               [-Gram, Gram]], Gram = K(X, X)
    grad = - [eps * ones(N) - y, eps * ones(N) + y]
    coef = zeros(2 * N)
    t = [ones(N), -ones(N)]

    Parameters
    ----------
    C : float (default: np.inf)
        inverse regularization parameter (inf - no penalty)
    kernel : str (default: "linear")
        kernel insted of inner product
    kernel_kwargs : kwargs
        parameters of kernel
    """
    def __init__(self, C=np.inf, eps=0.1, kernel="linear", **kernel_kwargs):
        if kernel not in kernels:
            raise ValueError(f"Unkown kernel: avilible only {list(kernels.keys())}")
        self.kernel = kernels[kernel](**kernel_kwargs)
        self.eps = eps
        super().__init__(C)

    def _fit(self, X, y, tol=1e-8):
        """
        estimate support vectors and their parameters
        Parameters
        ----------
        X : (N, D) np.ndarray
            training independent variable
        y : (N,) np.ndarray
            training dependent variable
            binary -1 or 1
        tol : float, optional
            numerical tolerance (the default is 1e-8)
        """
        N = len(y)
        coef = np.zeros(2 * N) # alphas(lagrangian multipliers)
        grad = -np.r_[self.eps * np.ones(N) - y, self.eps * np.ones(N) + y]
        t = np.r_[np.ones(N), -np.ones(N)]
        Gram = self.kernel(X, X)
        Q = np.c_[np.r_[Gram, -Gram], np.r_[-Gram, Gram]]
        alpha = super()._solve_qp(coef, grad, t, Q, tol)
        alpha = alpha[:N] - alpha[N:]
        support_mask = np.abs(alpha) > tol
        self.a = alpha[support_mask]
        self.X = X[support_mask]
        self.y = y[support_mask]

    def _predict(self, X):
        """
        predict values of the input

        Parameters
        ----------
        x : (sample_size, n_features) ndarray
            input

        Returns
        -------
        label : (sample_size,) ndarray
            predicted values
        """
        y = self.decision_function(X)
        return y

    def decision_function(self, X):
        """
        calculate target value

        Parameters
        ----------
        x : (sample_size, n_features) ndarray
            input

        Returns
        -------
        distance : (sample_size,) ndarray
            target value
        """
        target = np.sum(
            self.a
            * self.kernel(X, self.X),
            axis=-1) + self.b
        return target