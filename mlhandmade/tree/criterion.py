import numpy as np
import warnings

class BaseCriterion:
    def gain(self, y, splits):
        n = y.size
        n_left, n_right = splits[0].size, splits[1].size
        left, right = self(splits[0]), self(splits[1])
        return n_left / n * left + n_right / n * right

class ClassificationCriterion(BaseCriterion):
    def __init__(self, criterion):
        self.criterion = criterion
    
    def __call__(self, x):
        if self.criterion == "gini":
            return self.gini(x)
        elif self.criterion == "entropy":
            return self.entropy(x)
        else:
            raise ValueError("Got unknown criterion.")

    @staticmethod
    def entropy(x):
        # log2(p) raise a RuntimeWarning even with np.where()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = np.bincount(x) / x.size
            entrs = np.where(p != 0.0, -p * np.log2(p), 0.0)
            return np.sum(entrs)

    @staticmethod
    def gini(x):
        p = np.bincount(x) / x.size
        return np.sum(p * (1 - p))

class RegressionCriterion(BaseCriterion):
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, x):
        if self.criterion == "mse":
            return self.mse(x)
        elif self.criterion == "mae":
            return self.mae(x)
        else:
            raise ValueError("Got unknown criterion.")

    @staticmethod
    def mse(x):
        x_mean = np.mean(x)
        return np.mean((x - x_mean)**2)
    
    @staticmethod
    def mae(x):
        x_median = np.median(x)
        return np.mean(np.abs(x - x_median))