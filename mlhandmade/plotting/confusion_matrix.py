import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from ..model_selection import confusion_matrix

def plot_confusion_matrix(
    y_true,
    y_pred,
    *,
    include_values=True,
    cmap="viridis",
    xticks_rotation="horizontal",
    values_format=None,
    display_labels=None,
    ax=None,
    colorbar=True,
    im_kw=None,
    **cm_kwargs
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    cm = confusion_matrix(y_true, y_pred, **cm_kwargs)
    n_classes = cm.shape[0]
    
    default_im_kw = dict(interpolation="nearest", cmap=cmap)
    im_kw = im_kw or {}
    im_kw = {**default_im_kw, **im_kw}

    im = ax.imshow(cm, **im_kw)
    text = None
    cmap_min, cmap_max = im.cmap(0), im.cmap(1.0)

    if include_values:
        text = np.empty_like(cm, dtype=object)

        # `text` color is depends from color of background
        thresh = (cm.max() + cm.min()) / 2.0

        # cartesian product of two axes of confusion matrix
        for i, j in product(range(n_classes), range(n_classes)):
            color = cmap_max if cm[i, j] < thresh else cmap_min

            if values_format is None:
                text_cm = format(cm[i, j], ".2g")
                if cm.dtype.kind != "f":
                    text_d = format(cm[i, j], "d")
                    if len(text_d) < len(text_cm):
                        text_cm = text_d
            else:
                text_cm = format(cm[i, j], values_format)
            
            text[i, j] = ax.text(
                j, i, text_cm, ha="center", va="center", color=color
            )

    if display_labels is None:
        display_labels = np.arange(n_classes)
    
    if colorbar:
        fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=display_labels,
        yticklabels=display_labels,
        ylabel="True label",
        xlabel="Predicted label"
    )

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

    return ax