import numpy as np
import pytest
from graphix import Pattern, command
from graphix.fundamentals import Plane
from graphix.random_objects import rand_circuit
from graphix.sim.base_backend import Backend
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
    pattern.add(command.E(nodes=(0, 1)))
    pattern.add(command.M(node=0, plane=Plane.XY, angle=0.5))
    pattern.add(command.M(node=1, plane=Plane.XY, angle=0.4, s_domain={0}))
    # pattern1 = preprocess_pauli(pattern, leave_input=False)
    # pattern.perform_pauli_measurements()
    print(list(pattern))


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
    pattern1 = preprocess_pauli(pattern, leave_input=False)
    pattern.perform_pauli_measurements()
    print(list(pattern), list(pattern1))
