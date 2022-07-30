import numpy as np
from collections.abc import Mapping, MutableSequence
from itertools import product
from functools import partial, reduce
from operator import mul

from ..base import BaseEstimator
from ..model_selection import cross_val_score
from ..utils.validations import check_random_state

class ParamGrid:
    """
    ParamGrid class implements datastructure with:
    1. iterator over all parameter combinations in grid 
    2. len() overload to get number of combinations
    """
    def __init__(self, param_grid):
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]
        elif isinstance(param_grid, MutableSequence):
            pass
        else:
            raise ValueError(
                f"`params` must be a dict or a list, but got {type(param_grid).__name__}."
            )
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
    """
    Grid search via cross validation

    Parameters
    ----------
    estimator : BaseEstimator class
        subclass of BaseEstimator
        or class with `fit` and `predict` attributes
    param_grid : list or dict
        grid of parameters to score
    scoring : Callable (default: None)
        score metric
    cv : int (default: 5)
        number of cross validation folds
    shuffle : bool (default: False)
        shuffle data in KFoldCV if True
    random_state : None, int or np.random.RandomState (default: 0)
        random_state or seed
    score_kwargs : kwargs
        keyword arguments for scoring
    """
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
        self.estimator = self._validate_estimator(estimator)
        self.param_grid = ParamGrid(param_grid)
        self.scoring = scoring
        self.cv = cv
        self.shuffle = shuffle
        self.rgen = check_random_state(random_state)
        self.score_kwargs = score_kwargs
    
    @staticmethod
    def _validate_estimator(estimator):
        message = f"estimator {estimator.__name__} must have `fit` and `predict` attributes"
        if not hasattr(estimator, "fit"):
            raise TypeError(message)
        if not hasattr(estimator, "predict"):
            raise TypeError(message)
        return estimator

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
        """
        Get predictions from best estimator
        """
        return self.best_estimator_.predict(X)