import pytest
from numpy.random import PCG64, Generator

SEED = 25


@pytest.fixture()
def fx_bg() -> PCG64:
    return PCG64(SEED)


@pytest.fixture()
def fx_rng(fx_bg: PCG64) -> Generator:
    return Generator(fx_bg)
