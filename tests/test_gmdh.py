import pytest
import os
import sys
import numpy as np

#GMDH_ROOT = os.environ['GMDH_ROOT']
#GMDH_BINARY_FILES = os.environ['GMDH_BINARY_FILES']
#sys.path.append(GMDH_ROOT)
#sys.path.append(GMDH_BINARY_FILES)

import gmdh

"""@pytest.fixture(scope='class')
def sber_ts_data():
    return pd.read_csv("Sberbank.csv")['close'][-3000:]"""

class TestDataPreparations:

    @pytest.mark.parametrize('original_data, ts_lags',
                            [([1, 2, 3, 4, 5, 6], 1), 
                            ([1, 2, 3, 4, 5, 6], 4),
                            ([1, 2, 3, 4, 5, 6], 5),
                            ([np.random.random() for i in range(10000)], 50)])
    def test_time_series_transformation(self, original_data, ts_lags):
        """
        Testing gmdh.time_series_transformation() method using both small and large original_data.
        All values of the input argument are correct.
        """
        X, y = gmdh.time_series_transformation(original_data, lags=ts_lags)
        for i in range(len(original_data) - ts_lags):
            assert np.array_equal(X[i], original_data[i:i+ts_lags])
        assert np.array_equal(y, original_data[ts_lags:])

    @pytest.mark.parametrize('original_data, ts_lags',
                            [([1, 2, 3, 4, 5, 6], 0),  # lags = 0
                            ([1, 2, 3, 4, 5, 6], -1),  # lags < 0
                            ([1, 2, 3, 4, 5, 6], 6),  # lags = time series size
                            ([1, 2, 3, 4, 5, 6], 7),  # lags > time series size
                            ([], 2)])  # empty time series
    def test_time_series_transformation_value_error(self, original_data, ts_lags):
        """
        Testing gmdh.time_series_transformation() method using incorrect input arguments values.
        Expected result is ValueError.
        """
        with pytest.raises(ValueError) as err_info:
            X, y = gmdh.time_series_transformation(original_data, lags=ts_lags)

    @pytest.mark.parametrize('original_data, ts_lags',
                            [([1, 2, 3, 4, 5, 6], 3.5),  # not integer lags
                            ([1, 2, 3, 4, 5, 6], 'a'),  # not integer lags
                            (['a', 'b', 'c', 'd'], 3)])  # time series data contains strings
    def test_time_series_transformation_type_error(self, original_data, ts_lags):
        """
        Testing gmdh.time_series_transformation() method using incorrect input arguments types.
        Expected result is TypeError.
        """
        with pytest.raises(TypeError) as err_info:
            X, y = gmdh.time_series_transformation(original_data, lags=ts_lags)
    
    def test_split_data(self):
        """
        Testing gmdh.split_data() method using simple X and y arrays without shuffling.
        """
        X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        y = [10, 11, 12]
        x_train, x_test, y_train, y_test = gmdh.split_data(X, y, test_size=0.33)
        assert np.array_equal(x_train, [[1, 2, 3], [4, 5, 6]])
        assert np.array_equal(x_test, [[7, 8, 9]])
        assert np.array_equal(y_train, [10, 11])
        assert np.array_equal(y_test, [12])
    
    def test_split_data_with_shuffle(self):
        """
        Testing gmdh.split_data() method using simple X and y arrays with shuffling and the same random_state param 2 times.
        The expected resulting arrays for the first and second time should be equal.
        """
        X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        y = [10, 11, 12]
        x_train, x_test, y_train, y_test = gmdh.split_data(X, y, test_size=0.66, shuffle=True, random_state=1)
        x_train2, x_test2, y_train2, y_test2 = gmdh.split_data(X, y, test_size=0.66, shuffle=True, random_state=1)
        assert x_train.shape == (1, 3) and np.array_equal(x_train, x_train2)
        assert x_test.shape == (2, 3) and np.array_equal(x_test, x_test2)
        assert y_train.shape == (1,) and np.array_equal(y_train, y_train2)
        assert y_test.shape == (2,) and np.array_equal(y_test, y_test2)
    

    @pytest.mark.parametrize('X, y, test_size',
                            [([], [], 0.2),  # empty arrays
                            ([[1, 2, 3], [4, 5, 6]], [7, 8], 0.1),  # empty test arrays because of small test_size and small origin arrays
                            ([[1, 2, 3], [4, 5, 6]], [7, 8], 0.9),  # empty train arrays because of large test_size and small origin arrays
                            ([[1, 2, 3], [4, 5, 6]], [7], 0.5),  # different X rows and y size values
                            ([[1, 2, 3], [4, 5, 6]], [7, 8], 0),  # test_size = 0
                            ([[1, 2, 3], [4, 5, 6]], [7, 8], -1),  # test_size < 0
                            ([[1, 2, 3], [4, 5, 6]], [7, 8], 1),  # test_size = 1
                            ([[1, 2, 3], [4, 5, 6]], [7, 8], 10)])  # test_size > 1
    def test_split_data_value_error(self, X, y, test_size):
        """
        Testing gmdh.split_data() method using incorrect input arguments values.
        Expected result is ValueError.
        """
        with pytest.raises(ValueError) as err_info:
            x_train, x_test, y_train, y_test = gmdh.split_data(X, y, test_size=test_size)



