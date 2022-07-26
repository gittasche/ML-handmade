import numpy as np
from itertools import product
from functools import partial, reduce
from operator import mul

from ..base import BaseEstimator
from ..model_selection import cross_val_score
from ..utils.validations import check_random_state

class ParamGrid:
    def __init__(self, param_grid):
        if not isinstance(param_grid, (dict, list)):
            raise ValueError(
                f"`params` must be a dict or a list, but got {type(param_grid).__name__}."
            )
        if isinstance(param_grid, dict):
            param_grid = [param_grid]
        self.param_grid = param_grid

    def __iter__(self):
        for p in self.param_grid:
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        product = partial(reduce, mul)
        return sum(
            product(len(v) for v in p.values()) if p else 1 for p in self.param_grid
        )

class GridSearchCV(BaseEstimator):
    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        cv=5,
        shuffle=False,
        random_state=0,
        **score_kwargs
    ):
        self.estimator = estimator
        self.param_grid = ParamGrid(param_grid)
        self.scoring = scoring
        self.cv = cv
        self.shuffle = shuffle
        self.rgen = check_random_state(random_state)
        self.score_kwargs = score_kwargs
    
    def _fit(self, X, y):
        estimators = []
        params = []
        scores = np.zeros(len(self.param_grid))
        for i, param in enumerate(self.param_grid):
            estimator = self.estimator(**param)
            score = cross_val_score(
                estimator,
                X,
                y,
                score=self.scoring,
                cv=self.cv,
                shuffle=self.shuffle,
                random_state=self.rgen,
                **self.score_kwargs
            )
            estimators.append(estimator)
            params.append(param)
            scores[i] = score

        best_idx = np.argmax(scores)
        self.best_estimator_ = estimators[best_idx]
        self.best_params_ = params[best_idx]
        self.best_score_ = scores[best_idx]
        self.best_estimator_.fit(X, y)

    def _predict(self, X):
        return self.best_estimator_.predict(X)