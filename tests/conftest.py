import pytest

import gmdh


@pytest.fixture(autouse=True)
def add_gmdhpy(doctest_namespace):
    doctest_namespace["gmdh"] = gmdh