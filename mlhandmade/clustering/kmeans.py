import numpy as np
import numbers
from scipy.spatial.distance import cdist

from ..base import BaseEstimator
from ..utils.validations import check_random_state

class KMeans(BaseEstimator):
    """
    Lloyd k-means clustering algorirthm

    Parameters
    ----------
    n_clusters : int
        number of clusters
    init : ["random", "k-means++"] (default: "k-means++")
        way to initialize centroids
    max_iter : int (default: 300)
        number of iterations in Lloyd algorithm
    tol : float (default: 1e-4)
        tolerance value to check approximate convergence
    """
    def __init__(self, n_clusters, init="k-means++", max_iter=300, tol=1e-4, random_state=0):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.rgen = check_random_state(random_state)

        self.y_required = False

    def _validate_input(self):
        message = (
            "n_clusters must be int > 0,"
            f" got {self.n_clusters} of type {type(self.n_clusters).__name__}."
        )
        if not isinstance(self.n_clusters, numbers.Integral):
            raise ValueError(message)
        if not self.n_clusters > 0:
            raise ValueError(message)

        if self.init not in ["random", "k-means++"]:
            raise ValueError(
                "init must be \"random\" or \"k-means++\","
                f" got \"{self.init}\"."
            )

    def _fit(self, X, y=None):
        self._validate_input()
        X_mean = X.mean(axis=0)
        X -= X_mean

        centers_init = self._init_centroids(X)
        self.labels_, self.centers_ = self._lloyd_kmeans(X, centers_init)

    def _init_centroids(self, X):
        if self.init == "k-means++":
            centers, _ = self._kmeansplusplus(X)
        elif self.init == "random":
            seeds = self.rgen.permutation(self.n_samples)[:self.n_clusters]
            centers = X[seeds]
        
        return centers

    def _kmeansplusplus(self, X):
        centers = np.empty((self.n_clusters, self.n_features), dtype=X.dtype)

        center_id = self.rgen.randint(self.n_samples)
        indices = np.full(self.n_clusters, -1, dtype=int)
        centers[0] = X[center_id]
        indices[0] = center_id

        n_local_trials = 2 + int(np.log(self.n_clusters))
        closest_dist = cdist(centers[0, np.newaxis], X, metric="euclidean")
        current_pot = closest_dist.sum()

        for c in range(1, self.n_clusters):
            rand_vals = self.rgen.uniform(size=n_local_trials) * current_pot
            candidate_ids = np.searchsorted(np.cumsum(closest_dist), rand_vals)
            # ensure max(candidate_idx) > len(closest_dist)
            np.clip(candidate_ids, None, closest_dist.size - 1, out=candidate_ids)

            distances_to_candidates = cdist(X[candidate_ids], X, metric="euclidean")

            np.minimum(closest_dist, distances_to_candidates, out=distances_to_candidates)
            candidates_pot = distances_to_candidates.sum(axis=1)

            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist = distances_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            centers[c] = X[best_candidate]
            indices[c] = best_candidate
        
        return centers, indices

    def _lloyd_kmeans(self, X, centers_init):
        centers = centers_init
        labels = np.full(self.n_samples, -1, dtype=np.int32)

        for _ in range(self.max_iter):
            distances = cdist(centers, X, metric="euclidean")
            labels_new = np.argmin(distances, axis=0)
            centers_new = np.array([
                np.mean(X[labels_new == cl], axis=0) for cl in range(self.n_clusters)
            ])

            # check strict convergence
            if np.array_equal(labels, labels_new):
                break
            # check approximate convergence
            elif np.allclose(centers, centers_new, atol=self.tol):
                break

            centers = centers_new
            # assign using slice to avoid dtype change
            labels[:] = labels_new

        return labels, centers

    def _predict(self, X):
        distances = cdist(self.centers_, X)
        return np.argmin(distances, axis=0)