import pytest
import os
import sys

GMDH_ROOT = os.environ['GMDH_ROOT']
GMDH_BINARY_FILES = os.environ['GMDH_BINARY_FILES']
sys.path.append(GMDH_ROOT)
sys.path.append(GMDH_BINARY_FILES)

import gmdh


@pytest.fixture(autouse=True)
def add_gmdhpy(doctest_namespace):
    doctest_namespace["gmdh"] = gmdh