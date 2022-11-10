import pytest
import os
import sys

GMDH_ROOT = os.environ['GMDH_ROOT']
sys.path.append(GMDH_ROOT)


import gmdh


@pytest.fixture(autouse=True)
def add_gmdhpy(doctest_namespace):
    doctest_namespace["gmdh"] = gmdh