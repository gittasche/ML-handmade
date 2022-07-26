import numpy as np
import matplotlib.pyplot as plt

def scatter_plot(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ax = None,
    legend = 0,
    markers = ('s', 'x', 'o', '^', '*'),
    colors = ('IndianRed', 'RoyalBlue', 'lightgreen', 'grey', 'cyan')
):
    if ax is None:
        ax = plt.gca()

    for idx, cl in enumerate(np.unique(y)):
        ax.scatter( x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    s=40,
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, framealpha=0.3, scatterpoints=1, loc=legend)

    return ax