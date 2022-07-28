# ML-handmade

Implemented some ML routines including other ML stuff such as preprocessing, visualization and model selection.

### References

- Scikit-learn source code: [https://github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)
- PRML algorithms by ctgk: [https://github.com/ctgk/PRML](https://github.com/ctgk/PRML)
- Sokolov lectures on ML(RU): [https://github.com/esokolov/ml-course-hse](https://github.com/esokolov/ml-course-hse)
- ML handbook by Yandex SDA(RU): [https://ml-handbook.ru/](https://ml-handbook.ru/)
- MLAlgorithms by rushter: [https://github.com/rushter/MLAlgorithms](https://github.com/rushter/MLAlgorithms)
- mlxtend library by rasbt: [https://github.com/rasbt/mlxtend](https://github.com/rasbt/mlxtend)

### Algorithms implemented

* Linear models with different optmization methods(GD, SGD, Batch-SGD, SAG)
* KNN with three approaches(brute-force, kd-tree, ball-tree)
* Multiclass strategies (One-vs-One, One-vs-Rest)
* Support vector (SVC and $\epsilon$-SVR) with different kernels(Linear, RBF, Polynomial)
* Discriminant analysis(linear & quadratic) implemented using SVD
* Decision tree classifier and regressor
* Random forest classifier and regressor with bootstrap
* AdaBoost classifier and regressor
* Other ML stuff, for instance, k-fold cross validation, quality metrics, plotting, e.t.c

### Installation

It can be installed using `pip`

```bash
pip install mlhandmade
```