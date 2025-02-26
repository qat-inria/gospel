import numpy as np
import numpy.typing as npt
import pytest
from graphix import Pattern, command
from graphix.fundamentals import Plane
from graphix.random_objects import rand_circuit
from graphix.sim.base_backend import (
    Backend,
    FixedBranchSelector,
    RandomBranchSelector,
    State,
)
from graphix.sim.density_matrix import DensityMatrix
from graphix.sim.statevec import Statevec
from graphix.simulator import DefaultMeasureMethod
from numpy.random import PCG64, Generator

from stim_prepauli_preprocessing import preprocess_pauli


def fidelity(u: npt.NDArray[np.complex128], v: npt.NDArray[np.complex128]) -> float:
    return np.abs(np.dot(u.conjugate(), v))  # type: ignore[no-any-return]


def compare_backend_results(state1: State, state2: State) -> float:
    if isinstance(state1, Statevec) and isinstance(state2, Statevec):
        return fidelity(state1.flatten(), state2.flatten())
    if isinstance(state1, DensityMatrix):
        dm1 = state1
    elif isinstance(state1, Statevec):
        dm1 = DensityMatrix(state1)
    else:
        raise NotImplementedError
    if isinstance(state2, DensityMatrix):
        dm2 = state2
    elif isinstance(state2, Statevec):
        dm2 = DensityMatrix(state2)
    else:
        raise NotImplementedError
    return fidelity(dm1.rho.flatten(), dm2.rho.flatten())


def test_simple() -> None:
    pattern = Pattern()
    pattern.add(command.N(node=0))
    pattern.add(command.N(node=1))
    pattern.add(command.N(node=2))
    pattern.add(command.E(nodes=(0, 1)))
    pattern.add(command.E(nodes=(1, 2)))
    pattern.add(command.M(node=0, plane=Plane.XY, angle=0.5))
    pattern.add(command.M(node=1, plane=Plane.XY, angle=0.4, s_domain={0}))
    pattern2 = preprocess_pauli(pattern, leave_input=False)
    pattern.minimize_space()
    pattern2.minimize_space()
    backend = "statevector"
    # Simulating the unprocessed pattern with the measures chosen by stim
    pbs = FixedBranchSelector(pattern2.results, RandomBranchSelector())
    # Instantiate the measure method to retrieve the measures of the non-Pauli nodes
    measure_method = DefaultMeasureMethod()
    state = pattern.simulate_pattern(
        backend, branch_selector=pbs, measure_method=measure_method
    )
    # Simulating the processed pattern with the measures drawn for the previous simulation
    pbs2 = FixedBranchSelector(measure_method.results)
    state2 = pattern2.simulate_pattern(backend, branch_selector=pbs2)
    assert compare_backend_results(state2, state) == pytest.approx(1)


@pytest.mark.parametrize("jumps", range(1, 11))
@pytest.mark.parametrize("backend", ["statevector", "densitymatrix"])
# TODO: tensor network backend is excluded because "parallel preparation strategy does not support not-standardized pattern".
def test_pauli_measurement_random_circuit(
    fx_bg: PCG64, jumps: int, backend: Backend
) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 4
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize(method="mc")
    pattern.shift_signals(method="mc")
    pattern2 = preprocess_pauli(pattern, leave_input=False)
    pattern.minimize_space()
    pattern2.minimize_space()
    # Since the patterns are deterministic, we do not need to select a particular branch
    state = pattern.simulate_pattern(backend)
    state2 = pattern2.simulate_pattern(backend)
    assert compare_backend_results(state2, state) == pytest.approx(1)
