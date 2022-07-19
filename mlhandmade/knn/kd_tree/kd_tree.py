import numpy as np
from numpy.typing import ArrayLike

from .base import KDTreeBase

class KDTree(KDTreeBase):
    """
    Numpy based kd-tree.

    Parameters
    ----------
    points : array-like of shape (N, D)
        Array of points to build tree
    values : array-like of shape(N,)
        values of some function in points, 
        for instance binary classes or continuos value in case of regression
    """
    def __init__(self, points: ArrayLike, values: ArrayLike) -> None:
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        self.points = np.atleast_2d(points)
        super().__init__(self.points, values)
    
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
        
        if point.shape[0] != super().get_dim:
            raise ValueError(f"Point dimension {point.shape[1]} \
             must equal tree dimension {super().get_dim}")

        heap = []
        super()._query_knn(point, k, metric, heap)
        if sort:
            heap = sorted(heap)[::-1]
        
        values = np.array([h[3] for h in heap])
        distances = np.array([-h[0] for h in heap])
        
        return values, distances