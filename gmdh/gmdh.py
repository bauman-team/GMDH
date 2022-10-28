__all__ = [
    "time_series_transformation", 
    "split_data",
    "Solver",
    "PolynomialType",
    "CriterionType", 
    "Criterion",
    "ParallelCriterion",
    "SequentialCriterion"
]


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


class Criterion:
    """
    Class implementing calculations for all possible single external criterions of GMDH models.

    Parameters
    ----------
    criterion_type : gmdh.CriterionType
        Element from `gmdh.CriterionType` enumeration specifying the type of external criterion 
        that will be used to select the optimum solution during training GMDH model.
    solver : gmdh.Solver, default=gmdh.Solver.BALANCED
        Element from `gmdh.Solver` enumeration specifying the method of linear equations solving during training GMDH model.
    """
    def __init__(self, criterion_type, solver=Solver.BALANCED):
        if not isinstance(criterion_type, CriterionType):
            raise TypeError(f"{criterion_type} is not a 'CriterionType' type object")
        if not isinstance(solver, Solver):
            raise TypeError(f"{solver} is not a 'Solver' type object")

        self.__criterion = _gmdh_core.Criterion(_gmdh_core.CriterionType(criterion_type.value), _gmdh_core.Solver(solver.value))


class ParallelCriterion:
    """
    Class implementing calculations for double parallel external criterions of GMDH models.

    A double parallel external criterion is a combination of two weighted single criterions.
    The resulting value is calculated by the formula: result = alpha * criterion1 + (1 - alpha) * criterion2

    Parameters
    ----------
    criterion_type : gmdh.CriterionType
        Element from `gmdh.CriterionType` enumeration specifying the type of the first external criterion 
        that will be used to select the optimum solution during training GMDH model.
    second_criterion_type : gmdh.CriterionType
        Element from `gmdh.CriterionType` enumeration specifying the type of the second external criterion 
        that will be used to select the optimum solution during training GMDH model.
    alpha : float, default=0.5
        Contribution of the first criterion to the combined parallel criterion.
        The Contribution of the second criterion will be equal to (1 - alpha).
        Value must be in the (0, 1) range.
    solver : gmdh.Solver, default=gmdh.Solver.BALANCED
        Element from `gmdh.Solver` enumeration specifying the method of linear equations solving during training GMDH model.
    """
    def __init__(self, criterion_type, second_criterion_type, alpha=0.5, solver=Solver.BALANCED):
        if not isinstance(criterion_type, CriterionType):
            raise TypeError(f"{criterion_type} is not a 'CriterionType' type object")
        if not isinstance(second_criterion_type, CriterionType):
            raise TypeError(f"{second_criterion_type} is not a 'CriterionType' type object")
        if not isinstance(solver, Solver):
            raise TypeError(f"{solver} is not a 'Solver' type object")

        self.__criterion = _gmdh_core.ParallelCriterion(
            _gmdh_core.CriterionType(criterion_type.value), 
            _gmdh_core.CriterionType(second_criterion_type.value),
            alpha, _gmdh_core.Solver(solver.value))


class SequentialCriterion:
    """
    Class implementing calculations for double sequential external criterions of GMDH models.

    A double sequential external criterion is a combination of two single criterion that are applied one after the other.
    The first one is used to calculate criterion values for all solutions.
    The second one is used to recalculate criterion values for several solutions with the best values of the first criterion.

    Parameters
    ----------
    criterion_type : gmdh.CriterionType
        Element from `gmdh.CriterionType` enumeration specifying the type of the first external criterion 
        that will be used to select the optimum solution during training GMDH model.
    second_criterion_type : gmdh.CriterionType
        Element from `gmdh.CriterionType` enumeration specifying the type of the second external criterion 
        that will be used to recalcuate criterion values for solutions with the best first criterion values.
    solver : gmdh.Solver, default=gmdh.Solver.BALANCED
        Element from `gmdh.Solver` enumeration specifying the method of linear equations solving during training GMDH model.
    """
    def __init__(self, criterion_type, second_criterion_type, solver=Solver.BALANCED):
        if not isinstance(criterion_type, CriterionType):
            raise TypeError(f"{criterion_type} is not a 'CriterionType' type object")
        if not isinstance(second_criterion_type, CriterionType):
            raise TypeError(f"{second_criterion_type} is not a 'CriterionType' type object")
        if not isinstance(solver, Solver):
            raise TypeError(f"{solver} is not a 'Solver' type object")
        
        self.__criterion = _gmdh_core.SequentialCriterion(
            _gmdh_core.CriterionType(criterion_type.value), 
            _gmdh_core.CriterionType(second_criterion_type.value),
            _gmdh_core.Solver(solver.value))

"""
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
