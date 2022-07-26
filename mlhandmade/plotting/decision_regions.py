import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(
    X: np.ndarray,
    y: np.ndarray,
    classifier,
    *,
    ax = None,
    legend = 0,
    resolution = 0.02,
    margin = 0.25,
    markers = ('s', 'x', 'o', '^', '*'),
    colors = ('IndianRed', 'RoyalBlue', 'lightgreen', 'gray', 'cyan')
):
    if ax is None:
        ax = plt.gca()

    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    x2_min, x2_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    
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