# Group Method of Data Handling (GMDH) - the family of deep learning algorithms.

![][c++-shield] ![][boost-shield] ![][pybind-shield] ![][python-shield] ![][eigen-shield]

GMDH is a Python module that implements algorithms for the group method of data handling.

Read the Python module [documentation](https://bauman-team.github.io/GMDH/python/html/).
## About GMDH

GMDH is a machine learning Python module (API) based on C++ library for fast calculations. It realized the Group Method of Data Handling. It is a set of several algorithms for different machine learning tasks solution.

It was developed with a focus on providing fast experimentations and studing.

The gmdh module implements 4 popular varieties of algorithms from the family of GMDH algorithms (COMBI, MULTI, MIA, RIA), designed to solve problems of data approximation and time series prediction. The library also includes auxiliary functionality for basic data preprocessing and saving already trained models.

## Short theory

Group Method of Data Handling was applied in a great variety of areas for deep learning and knowledge discovery, forecasting and data mining, optimization and pattern recognition.
Inductive GMDH algorithms give possibility to find automatically interrelations in data, to select an optimal structure of model or network and to increase the accuracy of existing algorithms.

You can read the detailed theory at [gmdh.net](https://gmdh.net/index.html).

---

## Installation

To install gmdh package you need run command:

```
pip install gmdh
```
Using:
```python
import gmdh
```

---

## First contact with gmdh

Let's consider the simplest example of using the basic combinatorial COMBI algorithm from the gmdh module.

To begin with, we import the Combi model and the split_data function from the module to split the source data into training and test samples:
```python
from gmdh import Combi, split_data
```

Let's create a simple dataset in which the target values of the matrix `y` will simply be the sum of the corresponding pair of values `x1` and `x2` of the matrix `X`:
```python
X = [[1, 2], [3, 2], [7, 0], [5, 5], [1, 4], [2, 6]]
y = [3, 5, 7, 10, 5, 8]
```

Let's divide our data into training and test samples:
```python
x_train, x_test, y_train, y_test = split_data(X, y)

# print result arrays
print('x_train:\n', x_train)
print('x_test:\n', x_test)
print('\ny_train:\n', y_train)
print('y_test:\n', y_test)
```
Output:
```
x_train:
 [[1. 2.]
 [3. 2.]
 [7. 0.]
 [5. 5.]
 [1. 4.]]
x_test:
 [[2. 6.]]

y_train:
 [ 3.  5.  7. 10.  5.]
y_test:
 [8.]
```

Let's create a `Combi` model, train it using training data by the `fit` method and then predict the result for the test sample using the `predict` method:
```python
model = Combi()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)

# compare predicted and real value
print('y_predicted: ', y_predicted)
print('y_test: ', y_test)
```

Output:
```
y_predicted:  [8.]
y_test:  [8.]
```

The predicted result coincided with the real value! Now we will output a polynomial that displays the pattern found by the model:
```python
model.get_best_polynomial()
```

Output:
```
'y = x1 + x2'
```

For more in-depth tutorials about gmdh you can check our online [GMDH_book](https://bauman-team.github.io/GMDH_book/intro.html).

---
## Documentation
Read the C++ library [documentation](https://bauman-team.github.io/GMDH/doxygen/html/).

Read the Python module [documentation](https://bauman-team.github.io/GMDH/python/html/).

---
## License

This project is licensed under the [Apache 2.0](https://github.com/bauman-team/GMDH/blob/master/LICENSE.md) License.

--- 
## Release notes

This is a bachelor's diploma project. It was written by students of the Bauman Moscow State Technical University (BMSTU). The first version 1.0.1 is released in PyPI. This version is the final one for the graduation project, but the project itself and the repository can continue to grow and improve. We will be glad to new ideas and suggestions.

All the release branches can be found on [GitHub](https://github.com/bauman-team/GMDH/releases).

All the release binaries can be found on [PyPI](https://pypi.org/project/gmdh/#history).

---
## Opening an issue and a PR

You can also post bug reports and feature requests in [GitHub issues](https://github.com/bauman-team/GMDH/issues). We welcome contributions!


[python-shield]:https://img.shields.io/badge/Make%20with-Python3-informational?style=flat&logo=python&logoColor=white&color=2bbc8a

[c++-shield]:https://img.shields.io/badge/Make%20with-C++14%20-informational?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH5wMPFxkOvYfa9gAAA7NJREFUOMttlF1oW2UYx3/POclJ0iTN1unGxlqtOHWsm8y6zWIL6kQvxvBjEwoKirihFzIdCAoi6C564cdAGGjVXQnChhfFodtwVoVt9ds2s7S0XVa7ZW2Tpm3SNicn57yvF0m6ZPW9ec95z3t+z/N/noe/8D/rQPdA5fFW4CBwCHCBD4HjwCzAZwe3rfhXql9e6u5HSkcB4DHgdaAd8JevOEAvcLS8OzdDl4HlrAxgO/Aa8CQQAdAro2eBk8DHQBzQFbBUyWsqy3sRWK8BrTWWaRC0DLQGu6goegoRqcAngO5yGZIAvjJsD9AFtACiNYSDJq3NMe69LcqaiIUGUlmHv8ez/JWYJ+8oRGgE3hXhqexS4Y2ddzb+IAe6B6JAD/AwgNawNmbR2baBLY0RDKkpM57S9I9nOdF3ncxCEcf1GJueYSiZOpVOTj/jAywgVqlVOGjS2baBrU1RAJTWLBY8BAgHfJiGcF9zDMdVdPUM8kciqa/P5UQptYpo2O+rjq61prU5xpbGCAALtsuZ/jQDE1kMhO3N9exuuYXL0/N8eu4SZ+MJbKdYklBWUgO0TINtTVEMEbSGM/1pvutPlRogMDqV42TfCD8PXSWTtUFrfIYQDVnMLTpoowqogYBlsCZqlbIruMQnchgiuK5LIp3R8avTMjO/QH2dxZHOHVxJ5Uikcuzf1cz7p+LEf0nUZrg8cIAgKK24OjXNPyNj+lomixcIgelHa03ecbGLHo6ryDsurqfANG4AhdKcpbIOGxuCRIIma4MFfrz4G/mlvCCC5Tk83rGV4bTNka//xFPgKcX5oUlydhHW1ddm6CrN72MZWhoj+E2DZx/aRGZujm/6RjAMYV/7Pbyyt5WpbIGj317i894hPCBXcEtN0boElHKdriQnde+FCWmw2tnXsZnV0RBvP9fBoad3IQLRkIWIEK0LsHNjhOO5WTCDGisgy12uj4QLw5fHM4NjCf5NTorrFDn8yVmU0jzx4N1YPpNYOLCswnZcTvw0yFtfnMPOLYKZF6wghMIz1EUcoWE/obYdj+RtuwuR+wEDpakPB9jzwF3sbdvEHetXo7Vm5NosPReGOf3rKIu2U5k9D7iIYbyJvXReePS9SvB1wAvAy8DtaEApTL9JXdAPGpbsIp7rgWFUrGcUOAZ8CaRrHakEFmAz8CrQCayqGacbt2fKkGPACADfv3OzxdWA/UAHcBjYDQTLX/PAaeAjoA9wKyBWxFwJpWwaz1MyXBf4APgKyFVnVb3+A4f0hvg6EphxAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDIzLTAzLTE1VDIzOjI1OjE0KzAyOjAwG0BtrgAAACV0RVh0ZGF0ZTptb2RpZnkAMjAyMy0wMy0xNVQyMzoyNToxNCswMjowMGod1RIAAAAASUVORK5CYII=&logoColor=white&color=2bbc8a

[boost-shield]:https://img.shields.io/badge/Make%20with-Boost-informational?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAIAAAAC64paAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAABmJLR0QA/wD/AP+gvaeTAAAAB3RJTUUH5wMPFwkLhy88KAAAAqpJREFUOMttVM9vnFUM9Nj+fm66SaUkLRGiEqAKEYkz4q/nwA0kxA2pElIJgjbQkKbZze77nj0cvt1VKuGTD555tt94QFLm2Cfch4gAAkAEAHZlh0TEHyNJZkZmgpkHLoHuwgAIecD7HsnMjIhaq4isJmYmAFN0SpLubpZmrqoHvM9vZmatdZomYfyzll+u7vpWzVRNX54NI6aIbBqfWVRtxruIkDtkKVuov3q7ghAiCsnMTWDQ2JQtZIBQRJoGgIqIijAiImopRZivb8rtqrhBRGrkydge42FVtaBbh64rSEaECIX0TEZErVGmSYWrAhGKSDKbpj1ry7sVX92wMTFTQi4vls+PkElVKJnzqiLi7sPdJ0c5dF0Q1OZ00Nw+XN0hahVSSBW+vllPKcIUEWUmMyMjIkqNWP97eY4vTvLFoujq+nrTvl8XN53/zVTXpf5+u1WApHNun4yo47h4c5v1/oOIqKqXdS6eAPMgkHkekU0NPhIJMjn27W/X8f3PbxZDY26i9vWnT894ezye32+2jc8kdNMXJ31mqKnO6nPTQPfTr3/3LRqTxsQRVzdB6571W6iXyJrc1Lw4GQaTJAA4AEDGof/hxz/f32+Oj9qkKMVMVw/bP+6efHVevzwdg+Ku7vZ8YZFsXEXEZ9Ga5ti7fBwku9ZK2SDeLsexs27oelDVOlWIiAJwtwh+e/ns7Ok41VQVQCJzueg/P7XI8LYXGNQJM3Mzm49MRWBmUB9a/e6bi1JzqqzBUnn52dJR+34xDsM4Dn3XtU3j7qo6Lx/M3B/GlJl/vVvXSEBUsexlmiZ3MzM3b9y9cbNGVTmfOMn9MWetVcGIyIjIrJGAAjBTUzOfG9aDJWDnJB+bQSbJnAuAnRnMRI/NBP9jQzvJzYrau9BeYXhkQ/8BmvvHTTs/0x0AAAAldEVYdGRhdGU6Y3JlYXRlADIwMjMtMDMtMTVUMjM6MDk6MTErMDI6MDALKBtGAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDIzLTAzLTE1VDIzOjA5OjExKzAyOjAwenWj+gAAAABJRU5ErkJggg==&logoColor=white&color=2bbc8a

[pybind-shield]:https://img.shields.io/badge/Make%20with-PyBind11-informational?style=flat&logo=data:image/png;base64,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&logoWidth=40&logoColor=white&color=2bbc8a

[eigen-shield]:https://img.shields.io/badge/Make%20with-Eigen-informational?style=flat&logo=data:image/png;base64,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&logoColor=white&color=2bbc8a