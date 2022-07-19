import numpy as np
from mlhandmade.svm.kernel import Kernel

class RBFKernel(Kernel):
    """
    Radial basis function kernel:
    k(x, y) = C0 * exp(-0.5 * (x - y).T @ P @ (x - y)),
    where P is a diagonal matrix of parameters.
    with `P = I / sigma^2` and `C0 = 0` we get classical
    representation from C. Bishop's PRML book

    Attributes
    ----------
    params : ndarray of shape (ndim + 1,)
        parameters of radial basis function
    ndim : int
        dimension of expected input data
    """
    def __init__(self, params):
        """
        construct Radial basis kernel function
        """
        assert params.ndim == 1
        self.params = params
        self.ndim = len(params) - 1

    def __call__(self, x, y, pairwise=True):
        """
        calculate radial basis function

        Parameters
        ----------
        x : ndarray [..., ndim]
            input of this kernel function
        y : ndarray [..., ndim]
            another input

        Returns
        -------
        output : ndarray
            output of this radial basis function
        """
        assert x.shape[-1] == self.ndim
        assert y.shape[-1] == self.ndim
        if pairwise:
            x, y = self._pairwise(x, y)
        d = self.params[1:] * (x - y) ** 2
        return self.params[0] * np.exp(-0.5 * np.sum(d, axis=-1))

    def derivatives(self, x, y, pairwise=True):
        if pairwise:
            x, y = self._pairwise(x, y)
        d = self.params[1:] * (x - y) ** 2
        delta = np.exp(-0.5 * np.sum(d, axis=-1))
        deltas = -0.5 * (x - y) ** 2 * (delta * self.params[0])[:, :, None]
        return np.concatenate((np.expand_dims(delta, 0), deltas.T))

    def update_parameters(self, updates):
        self.params += updates