import numpy as np
from scipy.stats import entropy

class BaseCriterion:
    def gain(self, y, splits, sample_weight):
        n = y.size
        n_left, n_right = splits[0].size, splits[1].size
        left, right = self(splits[0], sample_weight[0]), self(splits[1], sample_weight[1])
        return n_left / n * left + n_right / n * right

class ClassificationCriterion(BaseCriterion):
    def __init__(self, criterion):
        self.criterion = criterion
    
    def __call__(self, x, sample_weight):
        if self.criterion == "gini":
            return self.gini(x, sample_weight)
        elif self.criterion == "entropy":
            return self.entropy(x, sample_weight)
        else:
            raise ValueError("Got unknown criterion.")

    @staticmethod
    def entropy(x, sample_weight):
        p = np.bincount(x, weights=sample_weight) / np.sum(sample_weight)
        return entropy(p)

    @staticmethod
    def gini(x, sample_weight):
        p = np.bincount(x, weights=sample_weight) / np.sum(sample_weight)
        return 1 - np.sum(p**2)

class RegressionCriterion(BaseCriterion):
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, x, sample_weight):
        if self.criterion == "mse":
            return self.mse(x, sample_weight)
        elif self.criterion == "mae":
            return self.mae(x, sample_weight)
        else:
            raise ValueError("Got unknown criterion.")

    @staticmethod
    def mse(x, sample_weight):
        x_mean = np.average(x, weights=sample_weight)
        return np.average((x - x_mean)**2, weights=sample_weight)
    
    @staticmethod
    def mae(x, sample_weight):
        if sample_weight is None:
            x_median = np.median(x)
        else:
            x_median = np.median(x * sample_weight)
        return np.average(np.abs(x - x_median), weights=sample_weight)