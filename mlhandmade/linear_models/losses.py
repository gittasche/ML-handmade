from abc import abstractmethod
import numpy as np

class BaseLoss(object):
    def _set_params(self, X, y, w):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError("X is empty")

        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        if y.size == 0:
            raise ValueError("y is empty")

        if not isinstance(w, np.ndarray):
            w = np.array(w)

        if w.size == 0:
            raise ValueError("w is empty")

        self.X = X
        self.y = y
        self.w = w

    def loss(self, X, y, w):
        self._set_params(X, y, w)
        return self._loss(self.X, self.y, self.w)

    @abstractmethod
    def _loss(self, X, y, w):
        raise NotImplementedError()

    def grad(self, X, y, w):
        self._set_params(X, y, w)
        return self._grad(self.X, self.y, self.w)

    @abstractmethod
    def _grad(self, X, y, w):
        raise NotImplementedError()

class PerceptronLoss(BaseLoss):
    @staticmethod
    def _loss(X, y, w):
        z = y * (X @ w)
        return np.where(-z > 0, -z, 0)

    @staticmethod
    def _grad(X, y, w):
        z = y * (X @ w)
        der = np.where(-z > 0, -1, 0)
        if z.ndim == 0:
            return X * (y * der)
        else:
            return X * (y * der)[:, np.newaxis]

class LogisticLoss(BaseLoss):
    @staticmethod
    def _loss(X, y, w):
        z = y * (X @ w)
        return np.log(1 + np.exp(-z))
    
    @staticmethod
    def _grad(X, y, w):
        z = y * (X @ w)
        der = -np.exp(-z) / (1 + np.exp(-z))
        if z.ndim == 0:
            return X * (y * der)
        else:
            return X * (y * der)[:, np.newaxis]

class HingeLoss(BaseLoss):
    @staticmethod
    def _loss(X, y, w):
        z = y * (X @ w)
        return np.where(1 - z > 0, 1 - z, 0)

    @staticmethod
    def _grad(X, y, w):
        z = y * (X @ w)
        der = np.where(1 - z > 0, -1, 0)
        if z.ndim == 0:
            return X * (y * der)
        else:
            return X * (y * der)[:, np.newaxis]

class SigmoidLoss(BaseLoss):
    @staticmethod
    def _loss(X, y, w):
        z = y * (X @ w)
        return 2 / (1 + np.exp(z))
    
    @staticmethod
    def _grad(X, y, w):
        z = y * (X @ w)
        der = -2 * np.exp(z) / (1 + np.exp(z))**2
        if z.ndim == 0:
            return X * (y * der)
        else:
            return X * (y * der)[:, np.newaxis]

class ExpLoss(BaseLoss):
    @staticmethod
    def _loss(X, y, w):
        z = y * (X @ w)
        return np.exp(-z)

    @staticmethod
    def _grad(X, y, w):
        z = y * (X @ w)
        der = -np.exp(-z)
        if z.ndim == 0:
            return X * (y * der)
        else:
            return X * (y * der)[:, np.newaxis]