import numpy as np
from itertools import product
from ..utils.validations import check_consistent_length

def accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None):
    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)

    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)

    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must be equal size.")
    
    check_consistent_length(y_true, y_pred, sample_weight)

    score = y_true == y_pred

    if normalize:
        return np.average(score, weights=sample_weight)
    elif sample_weight is not None:
        return np.dot(score, sample_weight)
    else:
        return np.sum(score)

def confusion_matrix(y_true, y_pred, *, labels=None,  normalize=None, sample_weight=None):
    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)

    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)
    
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must be equal size.")

    if labels is None:
        labels = np.unique(y_true)
    else:
        labels = np.asarray(labels)
        n_labels = labels.size
        if n_labels == 0:
            raise ValueError("`labels` must contain at least one label")
        elif y_true.size == 0:
            return np.zeros((n_labels, n_labels), dtype=int)
        elif len(np.intersect1d(y_true, labels)) == 0:
            raise ValueError("At least one label specified in `labels` must be in `y_true`")
    
    if sample_weight is None:
        sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    else:
        sample_weight = np.asarray(sample_weight)

    check_consistent_length(y_true, y_pred, sample_weight)

    if normalize not in ["true", "pred", "all", None]:
        raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")

    n_labels = labels.size
    # `y_true` and `y_pred` should be indeces,
    # i.e 0, 1, 2, ..., so check if conversion
    # is needed
    need_index_conversion = not(
        labels.dtype.kind in {"i", "u", "b"} # dtype of `labels` must be int, uint or bool
        and np.all(labels == np.arange(n_labels)) # `labels` equals [0, 1, ..., n_labels - 1]
        and y_true.min() >= 0 # all elements of `y_true` are >= 0
        and y_pred.min() >= 0 # all elements of `y_pred` are >= 0
    )
    if need_index_conversion:
        # dict {label : label ordinal index}
        label_to_ind = {y: x for x, y in enumerate(labels)}
        # convert `y_pred` and `y_true` to indices corresponding to `labels`
        # if element not found in labels than index will be `n_labels + 1`
        y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])
        y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])

    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = np.logical_and(y_true < n_labels, y_pred < n_labels)
    if not np.all(ind):
        y_true = y_true[ind]
        y_pred = y_pred[ind]
        # `sample_weight` must be the same size as `y_true` and `y_pred`
        sample_weight = sample_weight[ind]

    if sample_weight.dtype.kind in {"i", "u", "b"}:
        dtype = np.int64
    else:
        dtype = np.float64

    # from scipy.sparse import coo_matrix
    # cm = coo_matrix(
    #     (sample_weight, (y_true, y_pred)),
    #     shape=(n_labels, n_labels),
    #     dtype=dtype,
    # ).toarray()

    cm = np.zeros(shape=(n_labels, n_labels), dtype=dtype)
    for w, t_true, t_pred in zip(sample_weight, y_true, y_pred):
        cm[t_true, t_pred] += w

    # context manager to avoid "same_kind" rule cast error
    # don't use `/=` operator for the same reason
    with np.errstate(all="ignore"):
        if normalize == "true":
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cm = cm / cm.sum()

    return cm