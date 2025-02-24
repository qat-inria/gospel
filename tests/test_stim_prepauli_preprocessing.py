import numpy as np
import pytest
from graphix import Pattern, command
from graphix.fundamentals import Plane
from graphix.random_objects import rand_circuit
from graphix.sim.base_backend import Backend, FixedBranchSelector
from graphix.sim.density_matrix import DensityMatrix
from graphix.sim.statevec import Statevec
from numpy.random import PCG64, Generator

from stim_prepauli_preprocessing import pattern_to_tableau_simulator, preprocess_pauli


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
    pattern2 = preprocess_pauli(pattern, leave_input=False)
    pattern.minimize_space()
    pattern2.minimize_space()
    pbs = FixedBranchSelector(pattern2.results)
    backend = "statevector"
    state = pattern.simulate_pattern(backend, branch_selector=pbs)
    state2 = pattern2.simulate_pattern(backend, branch_selector=pbs)
    assert compare_backend_result_with_statevec(
        backend, state2, state
    ) == pytest.approx(1)


def test_reduced() -> None:
    pattern = Pattern(input_nodes=[0, 1])
    pattern.extend(
        [
            command.N(node=2),
            command.N(node=3),
            command.N(node=4),
            command.E(nodes=(0, 2)),
            command.E(nodes=(1, 2)),
            command.E(nodes=(1, 4)),
            command.E(nodes=(2, 3)),
            command.M(node=0),
            command.M(node=1),
            command.M(node=2),
        ]
    )
    print("HERE")
    print(pattern_to_tableau_simulator(pattern, leave_input=False))
    pattern2 = preprocess_pauli(pattern, leave_input=False)
    pattern.minimize_space()
    pattern2.minimize_space()
    print(pattern2.results)
    print(list(pattern2))
    pbs = FixedBranchSelector(pattern2.results)
    backend = "statevector"
    state = pattern.simulate_pattern(backend, branch_selector=pbs)
    state2 = pattern2.simulate_pattern(backend, branch_selector=pbs)
    print(state)
    print(state2)
    assert compare_backend_result_with_statevec(
        backend, state2, state
    ) == pytest.approx(1)


def test_fixed() -> None:
    seq = [
        command.N(node=2),
        command.E(nodes=(0, 2)),
        command.M(node=0),
        command.E(nodes=(1, 2)),
        command.N(node=4),
        command.E(nodes=(1, 4)),
        command.M(node=1),
        command.N(node=3),
        command.E(nodes=(2, 3)),
        command.M(node=2),
        command.E(nodes=(4, 3)),
        command.N(node=5),
        command.E(nodes=(4, 5)),
        command.M(node=4),
        command.N(node=6),
        command.E(nodes=(5, 6)),
        command.M(node=5),
        command.N(node=7),
        command.E(nodes=(6, 7)),
        command.M(node=6),
        command.E(nodes=(7, 3)),
        command.N(node=8),
        command.E(nodes=(7, 8)),
        command.M(node=7),
        command.N(node=9),
        command.E(nodes=(8, 9)),
        command.M(node=8),
        command.N(node=10),
        command.E(nodes=(9, 10)),
        command.M(node=9),
        command.N(node=11),
        command.E(nodes=(10, 11)),
        command.M(node=10),
        command.E(nodes=(11, 3)),
        command.N(node=12),
        command.E(nodes=(11, 12)),
        command.M(node=11),
        command.N(node=13),
        command.E(nodes=(12, 13)),
        command.M(node=12),
        command.N(node=14),
        command.E(nodes=(3, 14)),
        command.M(node=3, angle=0.25, s_domain={2}),
        command.N(node=16),
        command.E(nodes=(13, 16)),
        command.M(node=13),
        command.N(node=17),
        command.E(nodes=(16, 17)),
        command.M(node=16, angle=0.25, s_domain={0, 1, 5, 7, 9, 11, 13}),
        command.N(node=15),
        command.E(nodes=(14, 15)),
        command.M(node=14),
        command.Z(node=17, domain={0, 1, 5, 7, 9, 11, 13}),
        command.Z(node=15, domain={8, 1, 10, 3}),
        command.X(node=15, domain={2, 14}),
        command.X(node=17, domain={16, 2, 4, 6, 8, 10, 12}),
    ]
    for l in range(len(seq)):
        print(l)
        pattern = Pattern(input_nodes=[0, 1])
        pattern.extend(seq[:l])
        pattern2 = preprocess_pauli(pattern, leave_input=False)
        pattern.minimize_space()
        pattern2.minimize_space()
        pbs = FixedBranchSelector(pattern2.results)
        backend = "statevector"
        state = pattern.simulate_pattern(backend, branch_selector=pbs)
        state2 = pattern2.simulate_pattern(backend, branch_selector=pbs)
        assert compare_backend_result_with_statevec(
            backend, state2, state
        ) == pytest.approx(1)


@pytest.mark.parametrize("jumps", range(1, 11))
@pytest.mark.parametrize("backend", ["statevector", "densitymatrix"])
# TODO: tensor network backend is excluded because "parallel preparation strategy does not support not-standardized pattern".
def test_pauli_measurement_random_circuit(
    fx_bg: PCG64, jumps: int, backend: Backend
) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 2
    depth = 1
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize(method="mc")
    pattern.shift_signals(method="mc")
    pattern2 = preprocess_pauli(pattern, leave_input=False)
    pattern.minimize_space()
    pattern2.minimize_space()
    pbs = FixedBranchSelector(pattern2.results)
    state = pattern.simulate_pattern(backend, branch_selector=pbs)
    state2 = pattern2.simulate_pattern(backend, branch_selector=pbs)
    print(list(pattern))
    assert compare_backend_result_with_statevec(
        backend, state2, state
    ) == pytest.approx(1)
