import numpy as np

from mlhandmade.base import BaseEstimator

def _class_means(X, y):
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, np.newaxis]
    return means

class LinearDiscriminantAnalysis(BaseEstimator):
    def __init__(self, tol=1e-4, n_components=None):
        self.tol = tol
        self.n_components = n_components

    def _solve_svd(self, X, y):
        n_samples, _ = X.shape
        n_classes = len(self.classes_)

        self.means_ = _class_means(X, y)

        Xc = []
        for idx, group in enumerate(self.classes_):
            Xg = X[y == group, :]
            Xc.append(Xg - self.means_[idx])
        
        self.xbar_ = np.dot(self.priors_, self.means_)

        Xc = np.concatenate(Xc, axis=0) # ravel in 2d case

        std = Xc.std(axis=0)
        std[std == 0] = 1.0 # avoid zero division
        fac = 1.0 / (n_samples - n_classes)

        # scaling of data
        X = np.sqrt(fac) * (Xc / std)

        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        rank = np.sum(S > self.tol)
        scalings = (Vt[:rank] / std).T / S[:rank]
        fac = 1.0 if n_classes == 1 else 1.0 / (n_classes - 1)

        X = np.dot(
            (
                (np.sqrt((n_samples * self.priors_) * fac))
                * (self.means_ - self.xbar_).T
            ).T,
            scalings
        )
        
        _, S, Vt = np.linalg.svd(X, full_matrices=False)

        rank = np.sum(S > self.tol * S[0])
        self.scalings_ = np.dot(scalings, Vt.T[:, :rank])
        coef = np.dot(self.means_ - self.xbar_, self.scalings_)
        self.intercept_ = -0.5 * np.sum(coef**2, axis=1) + np.log(self.priors_)
        self.coef_ = np.dot(coef, self.scalings_.T)
        self.intercept_ -= np.dot(self.xbar_, self.coef_.T)

    def _fit(self, X, y):
        self.classes_ = np.unique(y)

        # prior class probabilities from sample data
        _, y_t = np.unique(y, return_inverse=True)
        self.priors_ = np.bincount(y_t) / float(len(y))

        # maximum possible components to reduce
        max_components = min(len(self.classes_) - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        self._solve_svd(X, y)

        if self.classes_.size == 2:
            self.coef_ = np.array(
                self.coef_[1, :] - self.coef_[0, :], ndmin=2, dtype=X.dtype
            )
            self.intercept_ = np.array(
                self.intercept_[1] - self.intercept_[0], ndmin=1, dtype=X.dtype
            )

        self.n_features_out = self._max_components

    def _transform(self, X):
        X_new = np.dot(X - self.xbar_, self.scalings_)
        return X_new[:, :self._max_components]

    @staticmethod
    def _softmax(decisions):
        exp_a = np.exp(decisions - np.max(decisions, axis=-1, keepdims=True))
        return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

    def decision_function(self, X):
        scores = X @ self.coef_.T + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict_proba(self, X):
        decisions = self.decision_function(X)
        if self.classes_.size == 2:
            proba = 1.0 / (1 + np.exp(-decisions))
            return np.vstack([1 - proba, proba]).T
        else:
            return self._softmax(decisions)

    def _predict(self, X):
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = np.where(scores > 0, 1, 0)
        else:
            indices = np.argmax(scores, axis=-1)
        return self.classes_[indices]