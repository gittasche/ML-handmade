import numpy as np

def _num_samples(x):
    message = "Expected sequence or array-like, got %s" %type(x)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered a valid collection." % x)

        if isinstance(x.shape[0], int):
            return x.shape[0]

    try:
        return len(x)
    except TypeError:
        raise TypeError(message)

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
    if isinstance(seed, int):
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
    elif isinstance(sample_weight, (int, float)):
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