import numpy as np

from mlhandmade.base import BaseEstimator
from .linear import LinearKernel
from .poly import PolynomialKernel
from .rbf import RBFKernel

kernels = {
    "linear" : LinearKernel,
    "poly" : PolynomialKernel,
    "rbf" : RBFKernel
}

class KernelRidge(BaseEstimator):
    """
    Implementation of kernel ridge regression.
    Objective loss of ridge regression is a
    L(X, y, w) = ||Xw - y||^2_2 + 0.5 * alpha * ||w||^2_2
    and exact solution w = X^T @ (X @ X^T + alpha * I)^-1 @ y.
    Here we can make partial kernelization X @ X^T -> K(X, X).

    Parameters
    ----------
    alpha : float (default: 1.0)
        penalty parameter
    kernel : str (default: "linear")
        kernel instead of inner product
    kernel_kwargs : kwargs
        parameters of kernel
    """
    def __init__(self, alpha=1.0, kernel="linear", **kernel_kwargs):
        self.alpha = alpha
        self.kernel = kernels[kernel](**kernel_kwargs)

    def _fit(self, X, y):
        Gram = self.kernel(X, X)
        n_samples = Gram.shape[0]

        ravel = False
        if len(y.shape) == 1:
            y = y[:, np.newaxis]
            ravel = True
        
        Gram.flat[::n_samples + 1] += self.alpha

        try:
            self.dual_coef_ = np.linalg.solve(Gram, y)
        except np.linalg.LinAlgError:
            self.dual_coef_ = np.linalg.lstsq(Gram, y)[0]

        # put back Gram matrix
        Gram.flat[::n_samples + 1] -= self.alpha

        if ravel:
            self.dual_coef_ = self.dual_coef_.ravel()

        self.X_fit_ = X

    def _predict(self, X):
        Gram = self.kernel(X, self.X_fit_)
        return Gram @ self.dual_coef_