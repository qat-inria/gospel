import pytest
from numpy.random import PCG64, Generator

from gospel.sampling_circuits import get_circuit, ncircuits


@pytest.mark.parametrize("jumps", range(1, 11))
def test_sampled_circuit(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    index = rng.integers(ncircuits)
    circuit0 = get_circuit(index)
    circuit1 = get_circuit(index)
    assert circuit0.instruction == circuit1.instruction
