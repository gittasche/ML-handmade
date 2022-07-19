import numpy as np
import heapq
import itertools

class KDTreeBase:
    """
    kd-tree implementation class.

    Attributes
    ----------
    dim : ndarray of shape(D,)
        dimension of space
    division_axis : int
        splitting hyperplane orthoghonal to division axis
    data : ndarray of shape(D,)
        point in current leaf
    values : ndarray of shape(N,)
        values of some function in points, 
        for instance binary classes or continuos value in case of regression
    left, right : KDTreeBase object
        references to child leafs
        left and right doesn't make any sense, just analogy with binary tree
    """
    def __init__(self, points=None, values=None, division_axis=0):
        self.dim = points.shape[-1]
        self.division_axis = division_axis
        
        if points.shape[0] > 1:
            idx_sort = np.argsort(points[:, division_axis])
            points = points[idx_sort]
            values = values[idx_sort]

            # equivalent to `points.shape[0] // 2`
            middle = points.shape[0] >> 1
            division_axis = (division_axis + 1) % self.dim

            self.data = points[middle]
            self.value = values[middle]
            self.left = KDTreeBase(points[:middle], values[:middle], division_axis)
            self.right = KDTreeBase(points[middle + 1:], values[middle + 1:], division_axis)
        elif points.shape[0] == 1:
            self.data = points[0]
            self.value = values[0]
            self.left = None
            self.right = None
        else:
            self.data = None
            self.value = None

    def _query_knn(self, point, k, metric, heap, counter=itertools.count()):
        """
        k nearest neighbor query

        Parameters
        ----------
        point : ndarray of shape(D,)
            get neighbors of this point
        k : int
            number of neighbors
        metric : {1, 2, inf}
            metric of distance for numpy.linalg.norm function.
        heap : list
            heap of neighbors candidates,
            initial value suppose to be []
        counter : itertools.count object
            counter to avoid comparison error in heappushpop()
        """
        if self.data is not None:
            dist = self.distance(self.data, point, metric)
            dx = self.data[self.division_axis] - point[self.division_axis]
            # less distance must have higher priority,
            # so `-dist` in `item` instead of `dist`
            item = (-dist, next(counter), self.data, self.value)
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif dist < -heap[0][0]:
                heapq.heappushpop(heap, item)

            if dx < 0:
                if self.right is not None:
                    self.right._query_knn(point, k, metric, heap, counter)
            else:
                if self.left is not None:
                    self.left._query_knn(point, k, metric, heap, counter)
            if np.abs(dx) < -heap[0][0]:
                if dx >= 0:
                    if self.right is not None:
                        self.right._query_knn(point, k, metric, heap, counter)
                else:
                    if self.left is not None:
                        self.left._query_knn(point, k, metric, heap, counter)

    @staticmethod
    def distance(point1, point2, metric):
        return np.linalg.norm(point1 - point2, ord=metric)

    @property
    def get_dim(self):
        return self.dim