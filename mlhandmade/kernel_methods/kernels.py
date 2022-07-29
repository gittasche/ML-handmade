import numpy as np

def check_pairwise(X, Y):
    """
    Ensure arrays X and Y to 
    calculate Gram matrix
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "X and Y must have equal last dimension,"
            f" but {X.shape[1]} != {Y.shape[1]}"
        )

    return X, Y

class LinearKernel:
    """
    Linear kernel: K = X x Y^T + c

    Parameters
    ----------
    coef0 : float (default: 0.0)
        bias value
    """
    def __init__(self, coef0=0.0):
        self.coef0 = coef0

    def __call__(self, X, Y):
        X, Y = check_pairwise(X, Y)
        return X @ Y.T + self.coef0

class PolyKernel:
    """
    Polynomial kernel: K = (gamma * X x Y^T + c)^d

    Parameters
    ----------
    coef0 : float (default: 0.0)
        bias value
    degree : int (default: 3)
        degree of polynomial
    gamma : float (default: None)
        coefficient, by default `gamma = 1 / X.shape[1]`
    """
    def __init__(self, coef0=0.0, degree=3, gamma=None):
        self.coef0 = coef0
        self.degree = degree
        self.gamma = gamma

    def __call__(self, X, Y):
        X, Y = check_pairwise(X, Y)
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        return np.power(self.gamma * X @ Y.T + self.coef0, self.degree)

class RBFKernel:
    """
    RBF kernel: K = exp(-gamma * ||X - Y||^2)

    Parameters
    ----------
    gamma : float (default: None)
        coefficient, by default `gamma = 1 / X.shape[1]`
    """
    def __init__(self, gamma=None):
        self.gamma = gamma

    def __call__(self, X, Y):
        X, Y = check_pairwise(X, Y)
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]

        K = self._euclidean_sq_distances(X, Y)
        K *= -self.gamma
        np.exp(K, out=K)
        return K

    @staticmethod
    def _euclidean_sq_distances(X, Y):
        # einsum("ij,ij->i", X, X) - squared row norms of X
        XX = np.einsum("ij,ij->i", X, X)[:, np.newaxis]
        YY = np.einsum("ij,ij->i", Y, Y)[np.newaxis, :]

        distances = XX + YY - 2 * X @ Y.T
        np.maximum(distances, 0, out=distances)

        # prevent floating point rounding error
        if X is Y:
            np.fill_diagonal(distances, 0)

        return distances