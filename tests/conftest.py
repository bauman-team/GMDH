import pytest
import sys

sys.path.append("./build/Release") # execute from root repo
sys.path.append("./")

import gmdh


@pytest.fixture(autouse=True)
def add_gmdhpy(doctest_namespace):
    doctest_namespace["gmdh"] = gmdh