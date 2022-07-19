from abc import abstractmethod
import numpy as np

class BaseEstimator(object):
    """
    Base class for all estimators

    Attributes
    ----------
    y_required : bool (default: True)
        True if y is required for fit, it may be false for unsupervised
        learning algorithms
    fit_required : bool (default: True)
        True if fit required before predict
    """
    y_required = True
    fit_required = True

    def _set_params(self, X, y=None):
        """
        Params validation for estimator

        Parameters
        ----------
        X : array-like of shape (N, D)
            array of N objects with D features
        y : array-like of shape (N,)
            array of target variables
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        
        if X.size == 0:
            raise ValueError('X is an empty array.')
        
        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape[0]
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        if self.y_required:
            if y is None:
                raise ValueError('y is required for this estimator.')

            if not isinstance(y, np.ndarray):
                y = np.asarray(y)
            
            if y.size == 0:
                raise ValueError('y is an empty array.')

        return X, y
    
    def fit(self, X, y=None, **fit_params):
        X, y = self._set_params(X, y)
        self.fit_required = False
        return self._fit(X, y, **fit_params)

    @abstractmethod
    def _fit(self, X, y, **fit_params):
        raise NotImplementedError()

    def predict(self, X=None):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        
        if X.ndim != 2:
            X = np.atleast_2d(X)

        if X is not None and not self.fit_required:
            return self._predict(X)
        else:
            raise ValueError("Fit is necessary for this estimator.")
    
    @abstractmethod
    def _predict(self, X=None):
        raise NotImplementedError()