import numpy as np
from scipy import linalg

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
            self.dual_coef_ = linalg.solve(Gram, y, assume_a="pos", overwrite_a=False)
        except linalg.LinAlgError:
            self.dual_coef_ = linalg.lstsq(Gram, y)[0]

        # put back Gram matrix
        Gram.flat[::n_samples + 1] -= self.alpha

        if ravel:
            self.dual_coef_ = self.dual_coef_.ravel()

        self.X_fit_ = X

    def _predict(self, X):
        Gram = self.kernel(X, self.X_fit_)
        return Gram @ self.dual_coef_