import pytest
from numpy.random import PCG64

SEED = 25


@pytest.fixture()
def fx_bg() -> PCG64:
    return PCG64(SEED)
