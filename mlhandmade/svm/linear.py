import numpy as np

from mlhandmade.svm.kernel import Kernel

class LinearKernel(Kernel):
    """
    Linear kernel:
    k(x, y) = x @ y + c

    Parameters
    ----------
    const : float
        a constant to be added
    """
    def __init__(self, const=0.0):
        self.const = const

    def __call__(self, x, y, pairwise=True):
        """
        calculate pairwise linear kernel

        Parameters
        ----------
        x : (..., ndim) ndarray
            input
        y : (..., ndim) ndarray
            another input with the same shape
            
        Returns
        -------
        output : ndarray
            linear kernel
        """
        if pairwise:
            x, y = self._pairwise(x, y)
        return np.sum(x * y, axis=-1) + self.const