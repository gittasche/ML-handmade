import numpy as np
import heapq
import itertools

class BallTreeBase:
    """
    Implemnatation of ball-tree class.
    This implementation similar to sklearn binary tree,
    but with `leaf_size` = 1.
    Ball-tree contains more nodes than kd-tree, so it is a
    little bit slower, but still faster than brute force
    approach.

    Attributes:
    points : ndarray of shape (N, D)
        points of train dataset
    values : ndarray of shape(N,)
        values of some function in `points`,
        for example, continuos values in case of regression
    metric : int (default: 2)
        `ord` for `np.linalg.norm`
    """
    def __init__(self, points, values, metric=2):
        if points.shape[0] > 1:
            best_dimension = np.argmax(np.max(points, axis=0) - np.min(points, axis=0))
            idx_sort = np.argsort(points[:, best_dimension])
            points = points[idx_sort]
            values = values[idx_sort]

            middle = points.shape[0] >> 1
            self.data = np.mean(points, axis=0)
            self.value = values[middle]
            self.radius = np.max(np.linalg.norm(self.data - points, ord=metric, axis=-1))
            self.left = BallTreeBase(points[:middle], values[:middle], metric)
            self.right = BallTreeBase(points[middle:], values[middle:], metric)
        elif points.shape[0] == 1:
            self.data = points[0]
            self.value = values[0]
            self.left = None
            self.right = None
            self.radius = 0

    def _query_knn(self, point, k, metric, heap):
        """
        Function computes reduced distance between root and point
        RDist(node.data, point) = Dist(node.data, point) - node.radius
        """
        dist = self.distance(self.data, point, metric) - self.radius
        self._query_knn_reduced(point, k, metric, heap, dist)

    def _query_knn_reduced(self, point, k, metric, heap, rdist, counter=itertools.count()):
        if len(heap) and rdist >= -heap[0][0]:
            return
        elif self.left is None and self.right is None:
            # Case of leaf node: Dist(leaf.data, point) = RDist(leaf.data, point)
            item = (-rdist, next(counter), self.data, self.value)
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif rdist < -heap[0][0]:
                heapq.heappushpop(heap, item)
        else:
            left_rdist = self.distance(self.left.data, point, metric) - self.left.radius
            right_rdist = self.distance(self.right.data, point, metric) - self.right.radius
            if left_rdist <= right_rdist:
                self.left._query_knn_reduced(point, k, metric, heap, left_rdist, counter)
                self.right._query_knn_reduced(point, k, metric, heap, right_rdist, counter)
            else:
                self.right._query_knn_reduced(point, k, metric, heap, right_rdist, counter)
                self.left._query_knn_reduced(point, k, metric, heap, left_rdist, counter)
        return

    @staticmethod
    def distance(point1, point2, metric):
        return np.linalg.norm(point1 - point2, ord=metric)