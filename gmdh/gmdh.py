"""
GMDH: Group method of data handling
"""

# pylint: disable=c-extension-no-member, useless-parent-delegation, too-many-lines

import enum
import warnings
from abc import ABCMeta, abstractmethod
import numpy as np
from docstring_inheritance import NumpyDocstringInheritanceMeta  # pylint: disable=import-error
from gmdh import _gmdh_core

warnings.filterwarnings("always")

__all__ = [
    "time_series_transformation",
    "split_data",
    "Solver",
    "PolynomialType",
    "CriterionType",
    "Criterion",
    "ParallelCriterion",
    "SequentialCriterion",
    "Combi",
    "Multi",
    "Mia",
    "Ria",
    "FileError"
]

class FileError(Exception):
    """
    Exception class for file errors.
    """

class DocEnum(enum.Enum):
    """
    Enum class for adding docstring to elements of inherited enumerations.
    """
    def __new__(cls, value, doc=None):
        self = object.__new__(cls)
        self._value_ = value
        if doc is not None:  # pragma: no branch
            self.__doc__ = doc
        return self

class Solver(DocEnum):
    """
    Enumeration for specifying the method of linear equations solving in GMDH models.
    """
    FAST = _gmdh_core.Solver.FAST.value, "Fast solution with perhaps not the best accuracy"
    ACCURATE = _gmdh_core.Solver.ACCURATE.value, 'Slow solution with maximum accuracy'
    BALANCED = _gmdh_core.Solver.BALANCED.value, 'Balanced solution with average speed and accuracy'

class PolynomialType(DocEnum):
    """
    Enumeration for specifying the class of polynomials for GMDH models.

    It is the type of equations from which the set of solutions will be generated.
    Used for MIA and RIA models.
    """
    LINEAR = _gmdh_core.PolynomialType.LINEAR.value, \
        "Using linear equations: w0 + w1*x1 + w2*x2."
    LINEAR_COV = _gmdh_core.PolynomialType.LINEAR_COV.value, \
        "Using linear equations with covariation: w0 + w1*x1 + w2*x2 + w3*x1*x2."
    QUADRATIC = _gmdh_core.PolynomialType.QUADRATIC.value, \
        "Using quadratic equations: w0 + w1*x1 + w2*x2 + w3*x1*x2 + w4*x1^2 + w5*x2^2."

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
    ABSOLUTE_NOISE_IMMUNITY = _gmdh_core.CriterionType.ABSOLUTE_NOISE_IMMUNITY.value, ""
    SYM_ABSOLUTE_NOISE_IMMUNITY = _gmdh_core.CriterionType.SYM_ABSOLUTE_NOISE_IMMUNITY.value, ""


class Criterion:
    """
    Class implementing calculations for all possible single external criterions of GMDH models.

    Parameters
    ----------
    criterion_type : gmdh.CriterionType, default=gmdh.CriterionType.REGULARITY
        Element from `gmdh.CriterionType` enumeration specifying the type of external criterion
        that will be used to select the optimum solution during training GMDH model.
    solver : gmdh.Solver, default=gmdh.Solver.BALANCED
        Element from `gmdh.Solver` enumeration specifying the method of
        linear equations solving during training GMDH model.

    Attributes
    ----------
    criterion_type, solver : see Parameters

    Examples
    --------
    You can create `Criterion` object with parameters specifyied during initialization:

    >>> criterion = gmdh.Criterion(criterion_type=gmdh.CriterionType.STABILITY, \
solver=gmdh.Solver.FAST)

    Or you can create default `Criterion` object and then specify parameters using assignments:

    >>> criterion = gmdh.Criterion()
    >>> criterion.criterion_type = gmdh.CriterionType.STABILITY
    >>> criterion.solver = gmdh.Solver.FAST

    Displaying the current values of the arguments:

    >>> criterion = gmdh.Criterion()
    >>> criterion.criterion_type
    <CriterionType.REGULARITY: 0>
    >>> criterion.solver
    <Solver.BALANCED: 2>
    """
    def __init__(self, criterion_type=CriterionType.REGULARITY, solver=Solver.BALANCED):
        if not isinstance(criterion_type, CriterionType):
            raise TypeError(f"{criterion_type} is not a 'CriterionType' type object")
        if not isinstance(solver, Solver):
            raise TypeError(f"{solver} is not a 'Solver' type object")

        self._criterion_type = criterion_type
        self._solver = solver

    @property
    def criterion_type(self):  # pylint: disable=missing-function-docstring
        return self._criterion_type

    @criterion_type.setter
    def criterion_type(self, value):
        if not isinstance(value, CriterionType):
            raise TypeError(f"{value} is not a 'CriterionType' type object")
        self._criterion_type = value

    @property
    def solver(self):  # pylint: disable=missing-function-docstring
        return self._solver

    @solver.setter
    def solver(self, value):
        if not isinstance(value, Solver):
            raise TypeError(f"{value} is not a 'Solver' type object")
        self._solver = value

    def _get_core(self):
        return _gmdh_core.Criterion(
        _gmdh_core.CriterionType(self._criterion_type.value),
        _gmdh_core.Solver(self._solver.value))

class ParallelCriterion(Criterion):
    """
    Class implementing calculations for double parallel external criterions of GMDH models.

    A double parallel external criterion is a combination of two weighted single criterions.
    The resulting value is calculated by the formula:
    result = alpha * criterion1 + (1 - alpha) * criterion2.

    Parameters
    ----------
    criterion_type : gmdh.CriterionType, default=gmdh.CriterionType.REGULARITY
        Element from `gmdh.CriterionType` enumeration specifying the type of the first external
        criterion that will be used to select the optimum solution during training GMDH model.
    second_criterion_type : gmdh.CriterionType, default=gmdh.CriterionType.STABILITY
        Element from `gmdh.CriterionType` enumeration specifying the type of the second external
        criterion that will be used to select the optimum solution during training GMDH model.
    alpha : float, default=0.5
        Contribution of the first criterion to the combined parallel criterion.
        The Contribution of the second criterion will be equal to (1 - alpha).
        Value must be in the (0, 1) range.
    solver : gmdh.Solver, default=gmdh.Solver.BALANCED
        Element from `gmdh.Solver` enumeration specifying the method of
        linear equations solving during training GMDH model.

    Attributes
    ----------
    criterion_type, second_criterion_type, alpha, solver : see Parameters

    Examples
    --------
    You can create `ParallelCriterion` object with parameters specifyied during initialization:

    >>> criterion = gmdh.ParallelCriterion(criterion_type=gmdh.CriterionType.STABILITY, \
second_criterion_type=gmdh.CriterionType.REGULARITY, alpha=0.8, solver=gmdh.Solver.FAST)

    Or you can create default `ParallelCriterion` object
    and then specify parameters using assignments:

    >>> criterion = gmdh.ParallelCriterion()
    >>> criterion.criterion_type = gmdh.CriterionType.STABILITY
    >>> criterion.second_criterion_type = gmdh.CriterionType.REGULARITY
    >>> criterion.alpha = 0.8
    >>> criterion.solver = gmdh.Solver.FAST

    Displaying the current values of the arguments:

    >>> criterion = gmdh.ParallelCriterion()
    >>> criterion.criterion_type
    <CriterionType.REGULARITY: 0>
    >>> criterion.second_criterion_type
    <CriterionType.STABILITY: 2>
    >>> criterion.alpha
    0.5
    >>> criterion.solver
    <Solver.BALANCED: 2>
    """
    def __init__(self,
        criterion_type=CriterionType.REGULARITY,
        second_criterion_type=CriterionType.STABILITY,
        alpha=0.5, solver=Solver.BALANCED):

        super().__init__(criterion_type, solver)
        if not isinstance(second_criterion_type, CriterionType):
            raise TypeError(f"{second_criterion_type} is not a 'CriterionType' type object")
        if not isinstance(alpha, (int, float)):
            raise TypeError(f"{alpha} is not a float")
        if alpha >= 1 or alpha <= 0:
            raise ValueError("alpha value must be in the (0, 1) range")

        self._second_criterion_type = second_criterion_type
        self._alpha = alpha

    @property
    def second_criterion_type(self):  # pylint: disable=missing-function-docstring
        return self._second_criterion_type

    @second_criterion_type.setter
    def second_criterion_type(self, value):
        if not isinstance(value, CriterionType):
            raise TypeError(f"{value} is not a 'CriterionType' type object")
        self._second_criterion_type = value

    @property
    def alpha(self):  # pylint: disable=missing-function-docstring
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{value} is not a float")
        if value >= 1 or value <= 0:
            raise ValueError("alpha value must be in the (0, 1) range")
        self._alpha = value

    def _get_core(self):
        return _gmdh_core.ParallelCriterion(
            _gmdh_core.CriterionType(self._criterion_type.value),
            _gmdh_core.CriterionType(self._second_criterion_type.value),
            self._alpha, _gmdh_core.Solver(self._solver.value))


class SequentialCriterion(Criterion):
    """
    Class implementing calculations for double sequential external criterions of GMDH models.

    A double sequential external criterion is a combination of two single criterion
    that are applied one after the other. The first one is used to calculate criterion values
    for all solutions. The second one is used to recalculate criterion values for
    several solutions with the best values of the first criterion.

    Parameters
    ----------
    criterion_type : gmdh.CriterionType, default=gmdh.CriterionType.REGULARITY
        Element from `gmdh.CriterionType` enumeration specifying the type of the first external
        criterion that will be used to select the optimum solution during training GMDH model.
    second_criterion_type : gmdh.CriterionType, default=gmdh.CriterionType.STABILITY
        Element from `gmdh.CriterionType` enumeration specifying the type of the second external
        criterion that will be used to recalcuate criterion values for solutions
        with the best first criterion values.
    top : int, default=0
        The number of the best combinations that should be evaluated by the second criterion.
        If the parameter value equals to 0, about half of the combinations
        will remain for the second criterion.
    solver : gmdh.Solver, default=gmdh.Solver.BALANCED
        Element from `gmdh.Solver` enumeration specifying the method of
        linear equations solving during training GMDH model.

    Attributes
    ----------
    criterion_type, second_criterion_type, top, solver : see Parameters

    Examples
    --------
    You can create `SequentialCriterion` object with parameters specifyied during initialization:

    >>> criterion = gmdh.SequentialCriterion(criterion_type=gmdh.CriterionType.STABILITY, \
second_criterion_type=gmdh.CriterionType.REGULARITY, top=7, solver=gmdh.Solver.FAST)

    Or you can create default `SequentialCriterion` object
    and then specify parameters using assignments:

    >>> criterion = gmdh.SequentialCriterion()
    >>> criterion.criterion_type = gmdh.CriterionType.STABILITY
    >>> criterion.second_criterion_type = gmdh.CriterionType.REGULARITY
    >>> criterion.top = 7
    >>> criterion.solver = gmdh.Solver.FAST

    Displaying the current values of the arguments:

    >>> criterion = gmdh.SequentialCriterion()
    >>> criterion.criterion_type
    <CriterionType.REGULARITY: 0>
    >>> criterion.second_criterion_type
    <CriterionType.STABILITY: 2>
    >>> criterion.top
    0
    >>> criterion.solver
    <Solver.BALANCED: 2>
    """
    def __init__(self,
        criterion_type=CriterionType.REGULARITY,
        second_criterion_type=CriterionType.STABILITY,
        top=0,
        solver=Solver.BALANCED):

        super().__init__(criterion_type, solver)
        if not isinstance(second_criterion_type, CriterionType):
            raise TypeError(f"{second_criterion_type} is not a 'CriterionType' type object")
        if not isinstance(top, int):
            raise TypeError(f"{top} is not an 'int' type object")
        if top < 0:
            raise ValueError("top value must be a non-negative")

        self._second_criterion_type = second_criterion_type
        self._top = top

    @property
    def second_criterion_type(self):  # pylint: disable=missing-function-docstring
        return self._second_criterion_type

    @second_criterion_type.setter
    def second_criterion_type(self, value):
        if not isinstance(value, CriterionType):
            raise TypeError(f"{value} is not a 'CriterionType' type object")
        self._second_criterion_type = value

    @property
    def top(self):  #pylint: disable=missing-function-docstring
        return self._top

    @top.setter
    def top(self, value):
        if not isinstance(value, int):
            raise TypeError(f"{value} is not an 'int' type object")
        if value < 0:
            raise ValueError("top value must be a non-negative")
        self._top = value

    def _get_core(self):
        return  _gmdh_core.SequentialCriterion(
            _gmdh_core.CriterionType(self._criterion_type.value),
            _gmdh_core.CriterionType(self._second_criterion_type.value),
            self._top,
            _gmdh_core.Solver(self._solver.value))

class Meta(ABCMeta, NumpyDocstringInheritanceMeta):
    """
    Meta class from which the abstract Model class is inherited
    to provide posiibility avoiding writing duplicated docstrings.
    """

class Model(metaclass=Meta):
    """
    Abstract class that provides model interfaces and
    the implementation of all methods except fit.
    The methods contain only docstring information, which is the same for all models.
    The different parts are written in the methods of the child classes.
    """
    def __init__(self, model):
        self._model = model

    @abstractmethod
    def fit(self, X, y):  # pylint: disable=invalid-name
        """
        Parameters
        ----------
        X : array_like
            2D array containing numeric training data.
            Rows are data samples, columns are features or lags.
        y : array_like
            1D array containg target numeric values for the training data.

        See Also
        --------
        predict : Using fitted model to make predictions.
        """

        if np.isnan(X).sum() > 0:
            raise ValueError('X array contains nan values')
        if np.isnan(y).sum() > 0:
            raise ValueError('y array contains nan values')

    def predict(self, X, lags=None):  # pylint: disable=invalid-name
        """
        Make predictions based on the optimal solution found during the fitting process.

        This method can be used for both regular datasets with features
        and time-series datasets. For time series it is necessary to specify `lags` parameter.

        Parameters
        ----------
        X : array_like
            1D or 2D array containing numeric test data.
            If 2D array is used then rows are data samples, columns are features.
            If 1D array is used then the array should contain time series values and
            the array size must be equal to the lags number of the used training data.
        lags : int, default=None
            If `X` represents the time series data then lags parameter is the number
            of sequential values that will be predicted. If `X` isn't the time series
            the lags parameter must be None.

        Returns
        -------
        predictions : array_like
            1D array containing precited values for given `X` data
        """
        if lags is None:
            return self._model.predict(X)
        return self._model.predict(X, lags)

    def get_best_polynomial(self):
        """
        Getting a string representation of the formula of the best polynomial.

        Using this method you can see the polynomial that was constructed
        during the fitting process and will be used for predictions.

        Returns
        -------
        polynomial : str
            Polynomial of the fitted model.
        """
        return self._model.get_best_polynomial()

    def save(self, path):
        """
        Saving fitted model to the file.

        Using this method tou can save all the necessary information about structure
        and parameters of the fitted model in JSON format.

        Parameters
        ----------
        path : str
            Path to the file to save the model.

        Raises
        ------
        FileError
            If the file can't be created or opened.
        
        See Also
        --------
        load : Loading pre-trained model.
        """
        try:
            self._model.save(path)
        except _gmdh_core.FileError as err:
            raise FileError(err.args[0]) from err
        return self

    def load(self, path):
        """
        Loading model from the file

        Using this method you can load a pre-trained model to make predictions without fitting.

        Parameters
        ----------
        path : str
            Path to the file for loading the model.

        Raises
        ------
        FileError
            If the file can't be opened or it is corrupted. 
            Also raises when trying to load file of another model.
        """
        try:
            self._model.load(path)
        except _gmdh_core.FileError as err:
            raise FileError(err.args[0]) from err
        return self

class Combi(Model):
    """
    Class implementing combinatorial GMDH algorithm.

    It is the basic GMDH algorithm that checks all possible combinations
    from simple to complex ones. Each combination is a linear function.\n
    Combinations at the first level: y = w0 + w1*x1.\n
    Combinations at the second level: y = w0 + w1*x1 + w2*x2.\n
    Combinations at the N-th level: y = w0 + w1*x1 + w2*x2 + ... + wn*xn.
    """
    def __init__(self):
        super().__init__(_gmdh_core.Combi())

    def fit(self, X, y, criterion=Criterion(CriterionType.REGULARITY), test_size=0.5,  # pylint: disable=invalid-name
        p_average=1, n_jobs=1, verbose=0, limit=0):
        """
        Fitting the Combi model to find the best solution.

        Using the input training `X` and `y` data the Combi model looks for the optimal solution
        from linear functions and stops if the errors start to grow.
        The training process can be configured for specific purposes
        using hyperparameters.

        Parameters
        ----------
        criterion : gmdh.Criterion, default=gmdh.Criterion(gmdh.CriterionType.REGULARITY)
            External criterion is a function or specific combination of functions
            for solutions evaluation and choosing the best one.
        test_size : float, default=0.5
            Proportion of the input data to include in the internal test set
            that will be used only to calculate external criterion value.
            Value must be in the (0, 1) range.
        p_average : int, default=1
            Specifying the number of the best combinations for calculation
            the mean error value at each level.
        n_jobs : int, default=1
            The number of threads that will be used for calculations.
            If n_jobs=-1 the maximum possible threads will be used.
        verbose : {0, 1}, default=0
            If verbose=1 then the progress bars and additional information
            will be displayed during the model fitting.
            If verbose=0 there will be no information to display.
        limit : float, default=0
            If the error value at the end of the level decreases by less then limit value
            compared to the previous level the training process will stop.

        Returns
        -------
        self : Combi
            Fitted model.
        """
        super().fit(X, y)
        self._model.fit(X, y, criterion._get_core(), test_size, p_average, n_jobs, verbose, limit)
        return self

    def predict(self, X, lags=None):  # pylint: disable=invalid-name
        """
        Examples
        --------
        Creating data for fitting the model.
        The `X` array contains pairs of numbers, and
        the `y` array contains the corresponding sums of two numbers:

        >>> X = [[1, 2], [3, 2], [7, 0], [5, 5], [1, 4], [2, 6]]
        >>> y = [sum(row) for row in X]  # [3, 5, 7, 10, 5, 8]
        >>> x_train, x_test, y_train, y_test = gmdh.split_data(X, y, test_size=0.33)
        >>> y_test
        array([5., 8.])

        Fitting the model and making predictions:

        >>> model = gmdh.Combi()
        >>> model.fit(x_train, y_train)  # doctest: +ELLIPSIS
        <gmdh.gmdh.Combi object at 0x...>
        >>> y_pred = model.predict(x_test)
        >>> y_pred
        array([5., 8.])
        """
        return super().predict(X, lags)

    def get_best_polynomial(self):
        """
        Examples
        --------

        Fitting the Combi model using Fibonacci series:

        >>> X, y = gmdh.time_series_transformation([1, 1, 2, 3, 5, 8, 13, 21], lags=2)
        >>> x_train, x_test, y_train, y_test = gmdh.split_data(X, y, test_size=0.25)
        >>> model = gmdh.Combi()
        >>> model.fit(x_train, y_train).predict(x_test)
        array([13., 21.])
        >>> model.get_best_polynomial()
        'y = x1 + x2'
        """
        return super().get_best_polynomial()

    def save(self, path):
        """
        Returns
        -------
        self : Combi
            Combi model.
        """
        return super().save(path)

    def load(self, path):
        """
        Returns
        -------
        self : Combi
            Combi model loaded from the file.

        Examples
        --------

        Teaching the model to sum 2 numbers, then making predictions
        and saving model to the JSON file:

        >>> model1 = gmdh.Combi()
        >>> model1.fit(X=[[0, 2], [7, 4], [5, 5], [9, 12]], y=[2, 11, 10, 21])  # doctest: +ELLIPSIS
        <gmdh.gmdh.Combi object at 0x...>
        >>> model1.predict([[4, 3], [1, 11]])
        array([ 7., 12.])
        >>> model1.save('model1.json')  # doctest: +ELLIPSIS
        <gmdh.gmdh.Combi object at 0x...>

        Loading pre-trained model and making predictions without fitting:

        >>> model2 = gmdh.Combi()
        >>> model2.load('model1.json')  # doctest: +ELLIPSIS
        <gmdh.gmdh.Combi object at 0x...>
        >>> model2.predict([[4, 3], [1, 11]])
        array([ 7., 12.])
        """
        return super().load(path)

class Multi(Model):
    """
    Class implementing combinatorial selection Multi algorithm.

    This algorithm is an improvement of Combi algorithm. It works much faster
    but doesn't check all possible combinations.
    The idea of Multi algorithm is to select several best combinations
    of the input features (or lags) at each level and then combine this combinations
    with one of the other unused features (or lags) at the next level.
    As in the Combi algorithm, each combination in Multi is a linear function.\n
    Let there be 4 features in total (x1, x2, x3, x4)
    and the best combinations at the first level are:\n
    y = f(x2) and y = f(x3), where f - linear polynomail.\n
    Then at the second level Multi will consider combinations:\n
    y = f(x2, x1)\n
    y = f(x2, x3)\n
    y = f(x2, x4)\n
    y = f(x3, x1)\n
    y = f(x3, x4)\n
    At each level, one new variable will be added to the best combinations.
    """
    def __init__(self):
        super().__init__(_gmdh_core.Multi())

    def fit(self, X, y, criterion=Criterion(CriterionType.REGULARITY), k_best=1, test_size=0.5,
        p_average=1, n_jobs=1, verbose=0, limit=0):
        """
        Fitting the Multi model to find the best solution.

        Using the input training `X` and `y` data the Multi model looks for the optimal solution
        from linear functions and stops if the errors start to grow.
        The training process can be configured for specific purposes
        using hyperparameters.

        Parameters
        ----------
        criterion : gmdh.Criterion, default=gmdh.Criterion(gmdh.CriterionType.REGULARITY)
            External criterion is a function or specific combination of functions
            for solutions evaluation and choosing the best one.
        k_best : int, default=1
            The number of best combinations at each level that will be combined
            with other unused features (or lags) at the next level.
        test_size : float, default=0.5
            Proportion of the input data to include in the internal test set
            that will be used only to calculate external criterion value.
            Value must be in the (0, 1) range.
        p_average : int, default=1
            Specifying the number of the best combinations for calculation
            the mean error value at each level.
        n_jobs : int, default=1
            The number of threads that will be used for calculations.
            If n_jobs=-1 the maximum possible threads will be used.
        verbose : {0, 1}, default=0
            If verbose=1 then the progress bars and additional information
            will be displayed during the model fitting.
            If verbose=0 there will be no information to display.
        limit : float, default=0
            If the error value at the end of the level decreases by less then limit value
            compared to the previous level the training process will stop.

        Returns
        -------
        self : Multi
            Fitted model.
        """
        super().fit(X, y)
        self._model.fit(X, y, criterion._get_core(), k_best, test_size, p_average,
            n_jobs, verbose, limit)
        return self

    def predict(self, X, lags=None):  # pylint: disable=invalid-name
        """
        Examples
        --------
        Creating data for fitting the model.
        The `X` array contains pairs of numbers, and
        the `y` array contains the corresponding sums of two numbers:

        >>> X = [[1, 2], [3, 2], [7, 0], [5, 5], [1, 4], [2, 6]]
        >>> y = [sum(row) for row in X]  # [3, 5, 7, 10, 5, 8]
        >>> x_train, x_test, y_train, y_test = gmdh.split_data(X, y, test_size=0.33)
        >>> y_test
        array([5., 8.])

        Fitting the model and making predictions:

        >>> model = gmdh.Multi()
        >>> model.fit(x_train, y_train)  # doctest: +ELLIPSIS
        <gmdh.gmdh.Multi object at 0x...>
        >>> y_pred = model.predict(x_test)
        >>> y_pred
        array([5., 8.])
        """
        return super().predict(X, lags)

    def get_best_polynomial(self):
        """
        Examples
        --------

        Fitting the Multi model using Fibonacci series:

        >>> X, y = gmdh.time_series_transformation([1, 1, 2, 3, 5, 8, 13, 21], lags=2)
        >>> x_train, x_test, y_train, y_test = gmdh.split_data(X, y, test_size=0.25)
        >>> model = gmdh.Multi()
        >>> model.fit(x_train, y_train).predict(x_test)
        array([13., 21.])
        >>> model.get_best_polynomial()
        'y = x1 + x2'
        """
        return super().get_best_polynomial()

    def save(self, path):
        """
        Returns
        -------
        self : Multi
            Multi model.
        """
        return super().save(path)

    def load(self, path):
        """
        Returns
        -------
        self : Multi
            Multi model loaded from the file.

        Examples
        --------

        Teaching the model to sum 2 numbers, then making predictions
        and saving model to the JSON file:

        >>> model1 = gmdh.Multi()
        >>> model1.fit(X=[[0, 2], [7, 4], [5, 5], [9, 12]], y=[2, 11, 10, 21])  # doctest: +ELLIPSIS
        <gmdh.gmdh.Multi object at 0x...>
        >>> model1.predict([[4, 3], [1, 11]])
        array([ 7., 12.])
        >>> model1.save('model1.json')  # doctest: +ELLIPSIS
        <gmdh.gmdh.Multi object at 0x...>

        Loading pre-trained model and making predictions without fitting:

        >>> model2 = gmdh.Multi()
        >>> model2.load('model1.json')  # doctest: +ELLIPSIS
        <gmdh.gmdh.Multi object at 0x...>
        >>> model2.predict([[4, 3], [1, 11]])
        array([ 7., 12.])
        """
        return super().load(path)

class Mia(Model):
    """
    Class implementing multilayered iterative Mia algorithm.

    In Mia algorithm combinations are polynomails f(xi, xj).
    These polynomials can be non-linear. The type of polynomials
    can be specifying using `polynomial_type` parameter.

    Several best combinations are selected at each level.
    Then еach model is considered as a new variable.
    Each pair of these new variables generates
    a new variable when moving to the next level.

    Let there be 4 features in total (x1, x2, x3, x4)
    and the best combinations at the first level are:\n
    y = f1(x1, x2)\n
    y = f2(x1, x3)\n
    y = f3(x2, x4)\n
    Then at the second level Mia will consider combinations:\n
    y = f4(f1, f2)\n
    y = f5(f1, f3)\n
    y = f6(f2, f3)\n
    At each level, new combinations are constructed from the best polynomials of the last level.
    """
    def __init__(self):
        super().__init__(_gmdh_core.Mia())

    def fit(self, X, y, criterion=Criterion(CriterionType.REGULARITY), k_best=3,
        polynomial_type=PolynomialType.QUADRATIC,
        test_size=0.5, p_average=1, n_jobs=1, verbose=0, limit=0):
        """
        Fitting the Mia model to find the best solution.

        Using the input training `X` and `y` data the Mia model looks for the optimal solution
        from polynomials constructed from `polynomial_type` basic polynomials
        and stops if the errors start to grow. The training process can be configured
        for specific purposes using hyperparameters.

        Parameters
        ----------
        criterion : gmdh.Criterion, default=gmdh.Criterion(gmdh.CriterionType.REGULARITY)
            External criterion is a function or specific combination of functions
            for solutions evaluation and choosing the best one.
        k_best : int, default=3
            The number of best combinations at each level that will be combined
            with other unused features (or lags) at the next level.
            The minimum allowed value is 3.
        test_size : float, default=0.5
            Proportion of the input data to include in the internal test set
            that will be used only to calculate external criterion value.
            Value must be in the (0, 1) range.
        polynomial_type : gmdh.PolynomialType, default=gmdh.PolynomialType.QUADRATIC
            Specifying the type of polynomials that will be used to construct the final polynomial.
        p_average : int, default=1
            Specifying the number of the best combinations for calculation
            the mean error value at each level.
        n_jobs : int, default=1
            The number of threads that will be used for calculations.
            If n_jobs=-1 the maximum possible threads will be used.
        verbose : {0, 1}, default=0
            If verbose=1 then the progress bars and additional information
            will be displayed during the model fitting.
            If verbose=0 there will be no information to display.
        limit : float, default=0
            If the error value at the end of the level decreases by less then limit value
            compared to the previous level the training process will stop.

        Returns
        -------
        self : Mia
            Fitted model.
        """
        super().fit(X, y)
        self._model.fit(X, y, criterion._get_core(), k_best,
            _gmdh_core.PolynomialType(polynomial_type.value), test_size,
            p_average, n_jobs, verbose, limit)
        return self

    def predict(self, X, lags=None):  # pylint: disable=invalid-name
        """
        Examples
        --------

        Time series data preparations:

        >>> X, y = gmdh.time_series_transformation([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], lags=3)
        >>> x_train, x_test, y_train, y_test = gmdh.split_data(X, y)
        >>> x_test
        array([[ 7.,  8.,  9.],
               [ 8.,  9., 10.]])
        >>> y_test
        array([10., 11.])

        Fitting the model and making predictions 5 steps ahead (lags=5)
        based only on the first row of `x_test` data:

        >>> model = gmdh.Mia()
        >>> y_pred = model.fit(x_train, y_train).predict(x_test[0], lags=5)
        >>> y_pred
        array([10., 11., 12., 13., 14.])
        """
        return super().predict(X, lags)

    def get_best_polynomial(self):
        """
        Examples
        --------

        Creating data for fitting the model.
        The `X` matrix contains array of 4 numbers and
        the `y` array contains results of the function
        f(x1, x2, x3, x4) = x1^2 + x4^2 + 2*x1*x4:

        >>> X = [[0, 1, 2, 3],
        ...      [3, 4, 3, 2],
        ...      [2, 5, 1, 0],
        ...      [1, 1, 5, 6],
        ...      [2, 3, 1, 4],
        ...      [2, 6, 0, 1],
        ...      [3, 4, 2, 5]]
        >>> y = [row[0]**2 + 2*row[0]*row[3] + row[3]**2 for row in X]  # [9, 25, 4, 49, 36, 9, 64]
        >>> x_train, x_test, y_train, y_test = gmdh.split_data(X, y)
        >>> y_test
        array([64.])

        Fitting the model, making predictions and printing the best polynomial:

        >>> model = gmdh.Mia()
        >>> model.fit(x_train, y_train)  # doctest: +ELLIPSIS
        <gmdh.gmdh.Mia object at 0x...>
        >>> y_pred = model.predict(x_test)
        >>> y_pred
        array([64.])
        >>> model.get_best_polynomial()
        'y = 2*x1*x4 + x1^2 + x4^2'
        """
        return super().get_best_polynomial()

    def save(self, path):
        """
        Returns
        -------
        self : Mia
            Mia model.
        """
        return super().save(path)

    def load(self, path):
        """
        Returns
        -------
        self : Mia
            Mia model loaded from the file.

        Examples
        --------

        Time series data preparations, fitting the Mia model,
        making predictions and saving model to the JSON file:

        >>> X, y = gmdh.time_series_transformation([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], lags=3)
        >>> x_train, x_test, y_train, y_test = gmdh.split_data(X, y)
        >>> y_test
        array([10., 11.])
        >>> model1 = gmdh.Mia()
        >>> y_pred = model1.fit(x_train, y_train).predict(x_test)
        >>> y_pred
        array([10., 11.])
        >>> model1.save("model1.json")  # doctest: +ELLIPSIS
        <gmdh.gmdh.Mia object at 0x...>

        Loading pre-trained model and making predictions without fitting:

        >>> model2 = gmdh.Mia()
        >>> model2.load('model1.json')  # doctest: +ELLIPSIS
        <gmdh.gmdh.Mia object at 0x...>
        >>> model2.predict(x_test)
        array([10., 11.])
        """
        return super().load(path)

class Ria(Model):
    """
    Class implementing relaxation iterative Ria algorithm.

    This algorithm is an improvement of Mia algorithm.
    Fewer combinations are considered at each level.
    Algorithm combinations are polynomails f(xi, xj).
    These polynomials can be non-linear. The type of polynomials
    can be specifying using `polynomial_type` parameter.

    Several best combinations are selected at each level.
    Then еach model is considered as a new variable and
    these new variables are combined with the original variables
    to generate new variables when moving to the next level.

    Let there be 3 features in total (x1, x2, x3)
    and the best combinations at the first level are:\n
    y = f1(x1, x2)\n
    y = f2(x1, x3)\n
    Then at the second level Ria will consider combinations:\n
    y = f3(f1, x1)\n
    y = f4(f1, x2)\n
    y = f5(f1, x3)\n
    y = f6(f2, x1)\n
    y = f7(f2, x2)\n
    y = f8(f2, x3)\n
    At each level, new combinations are constructed from the best polynomials
    of the last level and the original variables.
    """
    def __init__(self):
        super().__init__(_gmdh_core.Ria())

    def fit(self, X, y, criterion=Criterion(CriterionType.REGULARITY), k_best=1,
        polynomial_type=PolynomialType.QUADRATIC,
        test_size=0.5, p_average=1, n_jobs=1, verbose=0, limit=0):
        """
        Fitting the Ria model to find the best solution.

        Using the input training `X` and `y` data the Ria model looks for the optimal solution
        from polynomials constructed from `polynomial_type` basic polynomials
        and stops if the errors start to grow. The training process can be configured
        for specific purposes using hyperparameters.

        Parameters
        ----------
        criterion : gmdh.Criterion, default=gmdh.Criterion(gmdh.CriterionType.REGULARITY)
            External criterion is a function or specific combination of functions
            for solutions evaluation and choosing the best one.
        k_best : int, default=1
            The number of best combinations at each level that will be combined
            with other unused features (or lags) at the next level.
        test_size : float, default=0.5
            Proportion of the input data to include in the internal test set
            that will be used only to calculate external criterion value.
            Value must be in the (0, 1) range.
        polynomial_type : gmdh.PolynomialType, default=gmdh.PolynomialType.QUADRATIC
            Specifying the type of polynomials that will be used to construct the final polynomial.
        p_average : int, default=1
            Specifying the number of the best combinations for calculation
            the mean error value at each level.
        n_jobs : int, default=1
            The number of threads that will be used for calculations.
            If n_jobs=-1 the maximum possible threads will be used.
        verbose : {0, 1}, default=0
            If verbose=1 then the progress bars and additional information
            will be displayed during the model fitting.
            If verbose=0 there will be no information to display.
        limit : float, default=0
            If the error value at the end of the level decreases by less then limit value
            compared to the previous level the training process will stop.

        Returns
        -------
        self : Ria
            Fitted model.
        """
        super().fit(X, y)
        self._model.fit(X, y, criterion._get_core(), k_best,
            _gmdh_core.PolynomialType(polynomial_type.value), test_size,
            p_average, n_jobs, verbose, limit)
        return self

    def predict(self, X, lags=None):  # pylint: disable=invalid-name
        """
        Examples
        --------

        Time series data preparations:

        >>> X, y = gmdh.time_series_transformation([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], lags=3)
        >>> x_train, x_test, y_train, y_test = gmdh.split_data(X, y)
        >>> x_test
        array([[ 7.,  8.,  9.],
               [ 8.,  9., 10.]])
        >>> y_test
        array([10., 11.])

        Fitting the model and making predictions 5 steps ahead (lags=5)
        based only on the first row of `x_test` data:

        >>> model = gmdh.Ria()
        >>> y_pred = model.fit(x_train, y_train).predict(x_test[0], lags=5)
        >>> y_pred
        array([10., 11., 12., 13., 14.])
        """
        return super().predict(X, lags)

    def get_best_polynomial(self):
        """
        Examples
        --------

        Creating data for fitting the model.
        The `X` matrix contains array of 3 numbers and
        the `y` array contains results of the function
        f(x1, x2, x3) = x1^2 + 10*x2^2 + 80:

        >>> X = [[0, 1, 1],
        ...      [3, 5, 4],
        ...      [2, 1, 3],
        ...      [1, 1, 4],
        ...      [2, 2, 1],
        ...      [2, 6, 0],
        ...      [3, 3, 4]]
        >>> y = [row[0]**2 + 10 * row[1]**2 + 80 for row in X]  # [90, 339, 94, 91, 124, 444, 179]
        >>> x_train, x_test, y_train, y_test = gmdh.split_data(X, y)
        >>> y_test
        array([179.])

        Fitting the model, making predictions and printing the best polynomial:

        >>> model = gmdh.Ria()
        >>> model.fit(x_train, y_train)  # doctest: +ELLIPSIS
        <gmdh.gmdh.Ria object at 0x...>
        >>> y_pred = model.predict(x_test)
        >>> y_pred
        array([179.])
        >>> model.get_best_polynomial()
        'y = x1^2 + 10*x2^2 + 80'
        """
        return super().get_best_polynomial()

    def save(self, path):
        """
        Returns
        -------
        self : Ria
            Ria model.
        """
        return super().save(path)

    def load(self, path):
        """
        Returns
        -------
        self : Ria
            Ria model loaded from the file.

        Examples
        --------

        Time series data preparations, fitting the Ria model,
        making predictions and saving model to the JSON file:

        >>> X, y = gmdh.time_series_transformation([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], lags=3)
        >>> x_train, x_test, y_train, y_test = gmdh.split_data(X, y)
        >>> y_test
        array([10., 11.])
        >>> model1 = gmdh.Ria()
        >>> y_pred = model1.fit(x_train, y_train).predict(x_test)
        >>> y_pred
        array([10., 11.])
        >>> model1.save("model1.json")  # doctest: +ELLIPSIS
        <gmdh.gmdh.Ria object at 0x...>

        Loading pre-trained model and making predictions without fitting:

        >>> model2 = gmdh.Ria()
        >>> model2.load('model1.json')  # doctest: +ELLIPSIS
        <gmdh.gmdh.Ria object at 0x...>
        >>> model2.predict(x_test)
        array([10., 11.])
        """
        return super().load(path)

def time_series_transformation(time_series, lags):
    """
    Converting a time series into `X` and `y` data.

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

def split_data(X, y, test_size=0.2, shuffle=False, random_state=0):  # pylint: disable=invalid-name
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
        If random_state=0, each time the data will be split randomly
        and the results may be different.

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

    Specifying the `test_size`:

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
