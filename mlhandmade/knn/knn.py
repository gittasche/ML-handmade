import numpy as np

from mlhandmade.base.base import BaseEstimator
from .kd_tree import KDTree
from .ball_tree import BallTree

algorithms = ["brute", "kd-tree", "ball-tree"]
metrics = {"l-1" : 1, "l-2" : 2, "l-inf" : np.inf}
weights_modes = ["uniform", "distance", "gaussian"]

class KNNBase(BaseEstimator):
    """
    Implementation of k Nearest Neighbors algorithm

    Attributes
    ----------
    k : int (default: 5)
        number of neighbors
    algorithm : str (default: "kd-tree")
        algorithm of neighbor search
        - "brute" : brute-force approach (no fit required)
        - "kd-tree" : kd-tree approach (fit required to build tree)
        - "ball-tree" : ball-tree approach (fit required to build tree)
        In case of tree algorithms using simple depth-first search.
        See `_query_single_depthfirst` in `sklearn/neighbors/_binary_tree.pxi`
    metric : str (default: "l-2")
        distance metric
        - "l-2": euclidian distance
        - "l-1": manhattan distance
        - "l-inf": max-abs distance
    weights-mode : str (default: "uniform")
        weights in decision based on distance
        - "uniform" : weights independent from distances
        - "distance" : weights = 1 / distances
        - "gaussian" : weights = exp(-2 * distances^2) / sqrt(2 * pi)
    """
    def __init__(self, k=5, algorithm="kd-tree", metric="l-2", weights_mode="uniform"):
        if k < 1:
            raise ValueError("k must be greater than 0.")
        if algorithm not in algorithms:
            raise ValueError(f"Unknown algorithm: availible only {algorithms}")
        if metric not in metrics:
            raise ValueError(f"Unknown metric: availible only {list(metrics.keys())}")
        if weights_mode not in weights_modes:
            raise ValueError(f"Unknown weights mode: availible only {weights_modes}")
        self.k = k
        self.algorithm = algorithm
        self.metric = metrics[metric]
        self.weights_mode = weights_mode

    def _fit(self, X, y):
        self.X = X
        self.y = y
        if self.algorithm == "kd-tree":
            self.tree = KDTree(X, y)
        elif self.algorithm == "ball-tree":
            self.tree = BallTree(X, y)
        elif self.algorithm == "brute":
            pass

    def _get_weights(self, distances):
        if self.weights_mode == "uniform":
            return None
        elif self.weights_mode == "distance":
            return 1 / distances
        elif self.weights_mode == "gaussian":
            return np.exp(-2 * distances**2) / np.sqrt(2 * np.pi)

    def _aggregate(self, neighbors_targets, weights):
        raise NotImplementedError()

    def _predict(self, X):
        return np.array([self._predict_x(x) for x in X])

    def _predict_x(self, x):
        if self.algorithm == "kd-tree":
            neighbors_targets, distances = self.tree.query_knn(x, self.k, metric=self.metric)
            weights = self._get_weights(distances)
        elif self.algorithm == "ball-tree":
            neighbors_targets, distances = self.tree.query_knn(x, self.k, metric=self.metric)
            weights = self._get_weights(distances)
        elif self.algorithm == "brute":
            distances = np.array([np.linalg.norm(x - example) for example in self.X])
            idx = np.argsort(distances)
            neighbors_targets = self.y[idx][:self.k]
            weights = self._get_weights(distances[idx][:self.k])
        
        return self._aggregate(neighbors_targets, weights)
        
class KNNClassifier(KNNBase):
    """
    Multinominal classification with KNN.
    Using np.bincount to get number of nearest class samples.
    """
    def _aggregate(self, neighbors_targets, weights):
        labels = np.bincount(neighbors_targets, weights=weights)
        return labels.argmax()


class KNNRegressor(KNNBase):
    """
    Regression with KNN.
    Using weighted average.
    """
    def _aggregate(self, neighbors_targets, weights):
        return np.average(neighbors_targets, weights=weights)