import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from mlhandmade.kernel_methods import (
    SupportVectorClassifier,
    SupportVectorRegressor,
    KernelRidge
)
from mlhandmade.multiclass import OVR, OVO
from mlhandmade.preprocessing import (
    standardize,
    binary,
    ordinal
)
from mlhandmade.plotting import(
    plot_decision_regions,
    scatter_plot,
)

iris = pd.read_csv("datasets/iris.csv")

X = iris.iloc[:100, [0, 1]].values
y = iris.iloc[:100, -1].values

X = standardize(X)
y = binary(y)

fig = plt.figure(figsize=(12,12))
model = SupportVectorClassifier(kernel="rbf", gamma=0.7, C=1.0)
model.fit(X, y)
ax = plot_decision_regions(X=X, y=y, classifier=model, legend=2)
ax.scatter(model.X[:, 0], model.X[:, 1], s=200, facecolor="none", edgecolor="g")
plt.show()