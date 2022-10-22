import pytest
import sys
import numpy as np

sys.path.append("C:/Users/Mi/Documents/Diploma/GMDH/build/Release")
sys.path.append("C:/Users/Mi/Documents/Diploma/GMDH")

sys.path.append("/home/mikhail-xnor/Projects/GMDH/build/Release")
sys.path.append("/home/mikhail-xnor/Projects/GMDH")

import gmdh

"""@pytest.fixture()
def simple_ts_data():
    return [1, 2, 3, 4, 5, 6]

@pytest.fixture(scope='class')
def sber_ts_data():
    return pd.read_csv("Sberbank.csv")['close'][-3000:]"""

class TestDataPreparations:

    @pytest.mark.parametrize('original_data, ts_lags',
                            [([1, 2, 3, 4, 5, 6], 1), 
                            ([1, 2, 3, 4, 5, 6], 4),
                            ([1, 2, 3, 4, 5, 6], 5),
                            ([np.random.random() for i in range(10000)], 50)])
    def test_time_series_transformation_on_simple_data(self, original_data, ts_lags):
        """
        Testing gmdh.time_series_transformation() method using both small and large original_data.
        All values of the input argument are correct.
        """
        x, y = gmdh.time_series_transformation(original_data, lags=ts_lags)
        for i in range(len(original_data) - ts_lags):
            assert np.array_equal(x[i], original_data[i:i+ts_lags])
        assert np.array_equal(y, original_data[ts_lags:])

    @pytest.mark.parametrize('original_data, ts_lags',
                            [([1, 2, 3, 4, 5, 6], 0), 
                            ([1, 2, 3, 4, 5, 6], -1),
                            ([1, 2, 3, 4, 5, 6], 6),
                            ([1, 2, 3, 4, 5, 6], 7),
                            ([], 2)])
    def test_time_series_transformation_value_error(self, original_data, ts_lags):
        """
        Testing gmdh.time_series_transformation() method using incorrect input arguments values.
        Expected result is ValueError.
        """
        with pytest.raises(ValueError) as err_info:
            x, y = gmdh.time_series_transformation(original_data, lags=ts_lags)

    @pytest.mark.parametrize('original_data, ts_lags',
                            [([1, 2, 3, 4, 5, 6], 3.5),
                            ([1, 2, 3, 4, 5, 6], 'a'),
                            (['a', 'b', 'c', 'd'], 3)])
    def test_time_series_transformation_type_error(self, original_data, ts_lags):
        """
        Testing gmdh.time_series_transformation() method using incorrect input arguments types.
        Expected result is TypeError.
        """
        with pytest.raises(TypeError) as err_info:
            x, y = gmdh.time_series_transformation(original_data, lags=ts_lags)

