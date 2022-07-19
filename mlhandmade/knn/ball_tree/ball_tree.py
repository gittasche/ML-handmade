import numpy as np
from numpy.typing import ArrayLike

from .base import BallTreeBase

class BallTree(BallTreeBase):
    """
    Numpy based ball-tree.

    Parameters
    ----------
    points : array-like of shape (N, D)
        Array of points to build tree
    values : array-like of shape(N,)
        values of some function in points, 
        for instance binary classes or continuos value in case of regression
    """
    def __init__(self, points: ArrayLike, values: ArrayLike, metric=2) -> None:
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        if not isinstance(values, np.ndarray):
            values = np.array(values)

        self.values = values
        self.points = np.atleast_2d(points)
        self.dim = points.shape[1]
        super().__init__(self.points, values, metric)

    def query_knn(
        self,
        point: ArrayLike,
        k: int,
        sort: bool = False,
        metric: int = 2
    ) -> np.ndarray:
        """
        k nearest neighbors query.

        Parameters
        ----------
        point : array-like of shape (D,)
            get neighbors of this point
        k : int
            number of neighbors
        sort : bool (default: False)
            sort neighbors by distance to point
        metric : {1, 2, numpy.inf}
            metric of distance for numpy.linalg.norm function.
            1 : l1 norm, sum(abs(x[i]))
            2 : l2 norm, sqrt(sum(x[i]**2))
            numpy.inf : l-inf norm, max(abs(x[i]))

        Returns
        -------
        ndarray of ndarray's of neighbors and distances if return_distances
        """
        if not isinstance(point, np.ndarray):
            point = np.array(point)
        
        if k < 1:
            raise ValueError("k must be greater than 0.")
        
        if point.shape[0] != self.dim:
            raise ValueError(f"Point dimension {point.shape[0]} \
             must equal tree dimension {self.dim}")

        # initialize heap with inf's to get all k neighbors
        # instead of unexpected n < k neighbors, due to
        # `distance(t, B.pivot) - B.radius ≥ distance(t, Q.first)`
        # condition
        heap = [(-np.inf, 0, 0, 0) for _ in range(k)]
        super()._query_knn(point, k, metric, heap)
        if sort:
            heap = sorted(heap)[::-1]
        
        values = np.array([h[3] for h in heap])
        distances = np.array([-h[0] for h in heap])
        
        return values, distances