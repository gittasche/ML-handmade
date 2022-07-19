import numpy as np
from copy import copy
from mlhandmade.base.base import BaseEstimator

class OVR(BaseEstimator):
    def __init__(self, classifier):
        if not hasattr(classifier, "decision_function"):
            raise ValueError("This estimator can not be used with OVR.")
        self.classifier = classifier
    
    def _fit(self, X: np.ndarray, y: np.ndarray):
        self.labels = np.unique(y)
        self.num = self.labels.shape[0]
        y_groups = np.array([np.where(y == label, 1, -1) for label in self.labels])
        self.classifiers_ = np.array([copy(self.classifier) for _ in range(self.num)])
        for cl, y_group in zip(self.classifiers_, y_groups):
            cl.fit(X, y_group)
    
    def _predict(self, X: np.ndarray):
        all_outputs = np.array([classifier.decision_function(X) for classifier in self.classifiers_])
        return np.argmax(all_outputs, axis=0)


class OVO(BaseEstimator):
    def __init__(self, classifier):
        if not hasattr(classifier, "decision_function"):
            raise ValueError("This estimator can not be used with OVO.")
        self.classifier = classifier

    def _fit(self, X: np.ndarray, y: np.ndarray):
        self.labels = np.unique(y)
        self.num = self.labels.shape[0]
        self.classifiers_ = np.array([copy(self.classifier) for _ in range((self.num * (self.num - 1)) // 2)])
        for i in range(self.num):
            for j in range(i+1, self.num):
                X_pair = X[(y == self.labels[i]) | (y == self.labels[j])]
                y_pair = y[(y == self.labels[i]) | (y == self.labels[j])]
                y_pair = np.where(y_pair == self.labels[i], 1, -1)
                self.classifiers_[i + j - 1].fit(X_pair, y_pair)
    
    def _predict(self, X: np.ndarray):
        votes = np.zeros((X.shape[0], self.num))
        confidences = np.zeros((X.shape[0], self.num))
        for i in range(self.num):
            for j in range(i+1, self.num):
                z = self.classifiers_[i + j - 1].predict(X)
                votes[z == 1, i] += 1
                votes[z == -1, j] += 1
                confidences[:, i] += z
                confidences[:, j] -= z
        transformed_confidences = confidences / (3 * (np.abs(confidences) + 1))
        total_votes = np.argmax(votes + transformed_confidences, axis=1)
        return total_votes