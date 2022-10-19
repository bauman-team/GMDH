__all__ = ["time_series_transformation"]


import gmdhpy


def time_series_transformation(x, lags):
    """
    Method for transformations of time series!!!

    Examples
    --------
    >>> x, y = gmdh.time_series_transformation([1, 2, 3, 4, 5, 6], lags=3)
    >>> x
    array([[1., 2., 3.],
           [2., 3., 4.],
           [3., 4., 5.]])
           
    """

    return gmdhpy.time_series_transformation(x, lags)
