import numpy as np
from mlhandmade.svm.kernel import Kernel

class PolynomialKernel(Kernel):
    """
    Polynomial kernel
    k(x, y) = (x @ y + c)^M

    Parameters
    ----------
    const : float
        a constant to be added
    degree : int
        degree of polynomial order
    """
    def __init__(self, degree=2, const=0.0):
        self.const = const
        self.degree = degree

    def __call__(self, x, y, pairwise=True):
        """
        calculate pairwise polynomial kernel

        Parameters
        ----------
        x : (..., ndim) ndarray
            input
        y : (..., ndim) ndarray
            another input with the same shape
            
        Returns
        -------
        output : ndarray
            polynomial kernel
        """
        if pairwise:
            x, y = self._pairwise(x, y)
        return (np.sum(x * y, axis=-1) + self.const) ** self.degree