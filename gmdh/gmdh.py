__all__ = ["time_series_transformation", "split_data", "Solver", "PolynomialType", "CriterionType"]


import _gmdh_core
import enum


class DocEnum(enum.Enum):
    def __new__(cls, value, doc=None):
        self = object.__new__(cls)
        self._value_ = value
        if doc is not None:
            self.__doc__ = doc
        return self

class Solver(DocEnum):
    """
    Enumeration for specifying the method of linear equations solving in GMDH models 
    """
    FAST = _gmdh_core.Solver.FAST.value, "Fast solving with low accuracy"
    ACCURATE = _gmdh_core.Solver.ACCURATE.value, 'Slow solving with high accuracy'
    BALANCED = _gmdh_core.Solver.BALANCED.value, 'Balanced solving with medium speed and accuracy'

class PolynomialType(DocEnum):
    """
    Enumeration for specifying the class of polynomials for GMDH models.

    It is the type of equations from which the set of solutions will be generated. 
    Used for MIA and RIA models.
    """
    LINEAR = _gmdh_core.PolynomialType.LINEAR.value, "Using linear equations: w0 + w1*x1 + w2*x2"
    LINEAR_COV = _gmdh_core.PolynomialType.LINEAR_COV.value, "Using linear equations with covariation: w0 + w1*x1 + w2*x2 + w3*x1*x2"
    QUADRATIC = _gmdh_core.PolynomialType.QUADRATIC.value, "Using quadratic equations: w0 + w1*x1 + w2*x2 + w3*x1*x2 + w4*x1^2 + w5*x2^2"

class CriterionType(DocEnum):
    """
    Enumeration for specitying the criterion to select the optimum solution.
    """
    REGULARITY = _gmdh_core.CriterionType.REGULARITY.value, ""
    SYM_REGULARITY = _gmdh_core.CriterionType.SYM_REGULARITY.value, ""
    STABILITY = _gmdh_core.CriterionType.STABILITY.value, ""
    SYM_STABILITY = _gmdh_core.CriterionType.SYM_STABILITY.value, ""
    UNBIASED_OUTPUTS = _gmdh_core.CriterionType.UNBIASED_OUTPUTS.value, ""
    SYM_UNBIASED_OUTPUTS = _gmdh_core.CriterionType.SYM_UNBIASED_OUTPUTS.value, ""
    UNBIASED_COEFFS = _gmdh_core.CriterionType.UNBIASED_COEFFS.value, ""
    ABSOLUTE_STABILITY = _gmdh_core.CriterionType.ABSOLUTE_STABILITY.value, ""
    SYM_ABSOLUTE_STABILITY = _gmdh_core.CriterionType.SYM_ABSOLUTE_STABILITY.value, ""

"""
class Criterion:
    def __init__(self, criterion_type, solver=Solver.BALANCED):
        pass

class Combi:
    def __init__(self):
        self.__combi = _gmdh_core.Combi()

    def fit(self, X, y, test_size=0.5, p_average=1, n_jobs=1, verbose=0, limit=0):
        self.__combi.fit(X, y, n_jobs=n_jobs, verbose=verbose)
        return self
"""


def time_series_transformation(time_series, lags):
    """
    Converting a time series into `X` and `y` data
    
    Using this method, you can transform one-dimensional array 
    with time series data into two arrays (`X` and `y` data) according
    to the required number of lags.
    
    Parameters
    ----------
    time_series : array_like
        1D array containing original time series.
    lags : int
        The number of lags. Acceptable values are from 1 to `time_series.size()` - 1.
        
    Returns
    -------
    data : tuple
        A tuple whose items are arrays of `X` and `y` data 
        constructed from a time series.

    Examples
    --------
    >>> x, y = gmdh.time_series_transformation([1, 2, 3, 4, 5, 6], lags=3)
    >>> x
    array([[1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.]])
    >>> y
    array([4., 5., 6.])
    """
    return _gmdh_core.time_series_transformation(time_series, lags)

def split_data(X, y, test_size=0.2, shuffle=False, random_state=0):
    """
    Splitting data into train and test subsets.

    Using this method, you can prepare your data for further work 
    with machine learning models by dividing it into training and test sets.
    The proportions of the subsets and the data shuffling 
    can be customized according to your choice.

    Parameters
    ----------
    X : array_like
        2D array containing numeric values. 
        Rows are data samples, columns are features or lags.
    y : array_lile
        1D array containg target numeric values.
    test_size : float, default=0.2
        Proportion of the data to include in the test set.
        Value must be in the (0, 1) range.
    shuffle : bool, default=False
        If shuffle=True, the data samples will be shuffled before splitting.
    random_state : int, default=0
        Specifying a nonzero integer to get the same split every time. 
        If random_state=0, each time the data will be split randomly and the results may be different.
        
    Returns
    -------
    splitting : list
        List containing four elements: `x_train`, `x_test`, `y_train` and `y_test` arrays.

    Examples
    --------
    >>> X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    >>> y = [3, 5, 7, 9, 11]
    >>> x_train, x_test, y_train, y_test = gmdh.split_data(X, y)
    >>> x_train, y_train
    (array([[1., 2.],
           [3., 4.],
           [5., 6.],
           [7., 8.]]), array([3., 5., 7., 9.]))
    >>> x_test, y_test
    (array([[ 9., 10.]]), array([11.]))

    >>> x_train, x_test, y_train, y_test = gmdh.split_data(X, y, test_size=0.6)
    >>> x_train, y_train
    (array([[1., 2.],
           [3., 4.]]), array([3., 5.]))
    >>> x_test, y_test
    (array([[ 5.,  6.],
           [ 7.,  8.],
           [ 9., 10.]]), array([ 7.,  9., 11.]))
    """
    data = _gmdh_core.split_data(X, y, test_size, shuffle, random_state)
    return [data.x_train, data.x_test, data.y_train, data.y_test]
