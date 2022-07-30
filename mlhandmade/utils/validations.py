import numpy as np
import numbers

def _num_samples(X):
    message = f"Expected sequence or array-like, got {type(X).__name__}"

    if not hasattr(X, "__len__") and not hasattr(X, "shape"):
        if hasattr(X, "__array__"):
            X = np.asarray(X)
        else:
            raise TypeError(message)

    if hasattr(X, "shape") and X.shape is not None:
        if len(X.shape) == 0:
            raise TypeError(f"Singleton array {X} cannot be considered a valid collection.")

        if isinstance(X.shape[0], numbers.Integral):
            return X.shape[0]

    try:
        return len(X)
    except Exception as err:
        raise TypeError(message) from err

def _num_features(X):
    message = f"Unable to find the number of features from X of type {type(X).__name__}"
    if not hasattr(X, "__len__") and not hasattr(X, "shape"):
        if not hasattr(X, "__array__"):
            raise TypeError(message)
        X = np.asarray(X)
    
    if hasattr(X, "shape"):
        if not hasattr(X.shape, "__len__") or len(X.shape) <= 1:
            message += f" with shape {X.shape}"
            raise TypeError(message)
        return X.shape[1]

    first_sample = X[0]

    if isinstance(first_sample, (str, bytes, dict)):
        message += f" where the samples are of type {type(first_sample).__name__}"
        raise TypeError(message)
    
    try:
        return len(first_sample)
    except Exception as err:
        raise TypeError(message) from err

def check_consistent_length(*arrays):
    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(l) for l in lengths]
        )

def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        f"{seed} can not be used for RandomState instance"
    )

def check_sample_weight(sample_weight, X):
    n_samples = _num_samples(X)

    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=np.float64)
    elif isinstance(sample_weight, (numbers.Integral, numbers.Real)):
        sample_weight = np.full(n_samples, sample_weight, dtype=np.float64)
    else:
        if not isinstance(sample_weight, np.ndarray):
            sample_weight = np.asarray(sample_weight)
        if sample_weight.ndim != 1:
            raise ValueError(
                f"sample_weights must be one dimensional array-like, got{sample_weight.ndim}"
            )
        if sample_weight.shape != (n_samples,):
            raise ValueError(
                f"sample_weights shape must be ({n_samples},), got {sample_weight.shape}"
            )

    return sample_weight