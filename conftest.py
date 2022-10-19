import pytest
import sys

sys.path.append("C:/Users/Mi/Documents/Diploma/GMDH/build/Release")
sys.path.append("C:/Users/Mi/Documents/Diploma/GMDH")

sys.path.append("/home/mikhail-xnor/Projects/GMDH/build/Release")
sys.path.append("/home/mikhail-xnor/Projects/GMDH")

import gmdh


@pytest.fixture(autouse=True)
def add_gmdhpy(doctest_namespace):
    doctest_namespace["gmdh"] = gmdh