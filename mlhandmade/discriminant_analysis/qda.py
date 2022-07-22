import numpy as np

from mlhandmade.base import BaseEstimator

class QuadraticDiscriminantAnalysis(BaseEstimator):
    def __init__(self, tol=1e-4, reg_param=0.0):
        self.tol = tol
        self.reg_param = reg_param

    def _fit(self, X, y):
        n_samples, _ = X.shape
        self.classes_, y = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)

        self.priors_ = np.bincount(y) / float(n_samples)
        means = []
        scalings = []
        rotations = []
        for ind in range(n_classes):
            Xg = X[y == ind, :]
            meang = np.mean(Xg, axis=0)
            means.append(meang)
            Xgc = Xg - meang
            _, S, Vt = np.linalg.svd(Xgc, full_matrices=False)

            S2 = S**2 / (len(Xg) - 1)
            S2 = ((1 - self.reg_param) * S2) + self.reg_param

            scalings.append(S2)
            rotations.append(Vt.T)
        self.means_ = np.asarray(means)
        self.scalings_ = np.asarray(scalings)
        self.rotations_ = np.asarray(rotations)

    def _decision_function(self, X):
        norm2 = []
        for i in range(len(self.classes_)):
            R = self.rotations_[i]
            S = self.scalings_[i]
            Xm = X - self.means_[i]
            X2 = np.dot(Xm, R * (S ** (-0.5)))
            norm2.append(np.sum(X2**2, axis=1))
        norm2 = np.asarray(norm2).T
        u = np.asarray([np.sum(np.log(s)) for s in self.scalings_])
        return -0.5 * (norm2 + u) + np.log(self.priors_)

    def decision_function(self, X):
        decisions = self._decision_function(X)
        if len(self.classes_) == 2:
            return decisions[:, 1] - decisions[:, 0]
        return decisions

    @staticmethod
    def _softmax(values):
        exp_a = np.exp(values - np.max(values, axis=-1, keepdims=True))
        return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

    def _predict(self, X=None):
        decisions = self._decision_function(X)
        y_pred = self.classes_.take(np.argmax(decisions, axis=1))
        return y_pred

    def predict_proba(self, X):
        values = self._decision_function(X)
        return self._softmax(values)