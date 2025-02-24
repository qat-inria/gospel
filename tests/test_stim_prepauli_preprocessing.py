import copy

import numpy as np
import pytest
from graphix import Pattern, command
from graphix.fundamentals import Plane
from graphix.random_objects import rand_circuit
from graphix.sim.base_backend import Backend, ConstBranchSelector, FixedBranchSelector
from graphix.sim.density_matrix import DensityMatrix
from graphix.sim.statevec import Statevec
from numpy.random import PCG64, Generator

from stim_prepauli_preprocessing import preprocess_pauli


def compare_backend_result_with_statevec(
    backend: str, backend_state, statevec: Statevec
) -> float:
    if backend == "statevector":
        return np.abs(np.dot(backend_state.flatten().conjugate(), statevec.flatten()))
    if backend == "densitymatrix":
        return np.abs(
            np.dot(
                backend_state.rho.flatten().conjugate(),
                DensityMatrix(statevec).rho.flatten(),
            )
        )
    raise NotImplementedError(backend)


def test_simple() -> None:
    pattern = Pattern()
    pattern.add(command.N(node=0))
    pattern.add(command.N(node=1))
    pattern.add(command.N(node=2))
    pattern.add(command.E(nodes=(0, 1)))
    pattern.add(command.E(nodes=(1, 2)))
    pattern.add(command.M(node=0, plane=Plane.XY, angle=0.5))
    pattern.add(command.M(node=1, plane=Plane.XY, angle=0.4, s_domain={0}))
    pattern1 = copy.deepcopy(pattern)
    pattern2 = preprocess_pauli(pattern1, leave_input=False)
    pattern1.perform_pauli_measurements()
    pattern.minimize_space()
    backend = "statevector"
    cbs = ConstBranchSelector(False)
    print(pattern2.results)
    print(list(pattern2))
    pbs = FixedBranchSelector(pattern2.results, cbs)
    state = pattern.simulate_pattern(backend, branch_selector=pbs)
    state1 = pattern1.simulate_pattern(backend, branch_selector=pbs)
    state2 = pattern2.simulate_pattern(backend, branch_selector=pbs)
    print(compare_backend_result_with_statevec("statevector", state1, state))
    print(compare_backend_result_with_statevec("statevector", state2, state))


@pytest.mark.parametrize("jumps", range(1, 11))
@pytest.mark.parametrize("backend", ["statevector", "densitymatrix"])
# TODO: tensor network backend is excluded because "parallel preparation strategy does not support not-standardized pattern".
def test_pauli_measurement_random_circuit(
    fx_bg: PCG64, jumps: int, backend: Backend
) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 3
    depth = 3
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize(method="mc")
    pattern.shift_signals(method="mc")
    pattern1 = copy.deepcopy(pattern)
    pattern2 = preprocess_pauli(pattern1, leave_input=False)
    pattern1.perform_pauli_measurements()
    pattern.minimize_space()
    state = pattern.simulate_pattern(backend, rng=rng)
    state1 = pattern1.simulate_pattern(backend, rng=rng)
    state2 = pattern2.simulate_pattern(backend, rng=rng)
    assert compare_backend_result_with_statevec(
        backend, state1, state
    ) == pytest.approx(1)
    assert compare_backend_result_with_statevec(
        backend, state2, state
    ) == pytest.approx(1)
