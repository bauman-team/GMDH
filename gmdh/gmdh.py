__all__ = ["time_series_transformation"]


import _gmdh_core


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
