import math

import pytest
from graphix.fundamentals import Plane
from graphix.measurements import Measurement
from graphix.sim.statevec import Statevec, StatevectorBackend
from graphix.states import BasicStates
from numpy.random import PCG64, Generator
from qiskit.quantum_info import Statevector  # type: ignore[attr-defined]

from gospel.brickwork_state_transpiler import transpile
from gospel.sampling_circuits import (
    circuit_to_qiskit,
    estimate_circuit_by_expectation_value,
    estimate_circuit_by_sampling,
    get_circuit,
    ncircuits,
    sample_truncated_circuit,
)
from gospel.scripts import fidelity


@pytest.mark.parametrize("jumps", range(1, 11))
def test_sample_truncated_circuit(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    _circuit = sample_truncated_circuit(nqubits=8, depth=10, rng=rng)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_sampled_circuit(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    index = rng.integers(ncircuits)
    circuit0 = get_circuit(index)
    circuit1 = get_circuit(index)
    assert circuit0.instruction == circuit1.instruction


@pytest.mark.parametrize("jumps", range(1, 11))
def test_estimate_circuit_vs_sampling(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    index = rng.integers(ncircuits)
    circuit = get_circuit(index)
    qc = circuit_to_qiskit(circuit)
    p1 = estimate_circuit_by_expectation_value(qc)
    p2 = estimate_circuit_by_sampling(qc, seed=rng.integers(2 << 16))
    assert math.isclose(p1, p2, abs_tol=1e-1)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_simulate_circuit_vs_qiskit(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    index = rng.integers(ncircuits)
    circuit = get_circuit(index)
    qc = circuit_to_qiskit(circuit)
    sv1 = circuit.simulate_statevector(input_state=BasicStates.ZERO).statevec
    sv2 = Statevector.from_instruction(qc)
    assert fidelity(
        sv1.psi.transpose(*reversed(range(len(sv1.psi.shape)))).flatten(), sv2.data
    ) == pytest.approx(1)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_simulate_pattern_vs_qiskit(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    index = rng.integers(ncircuits)
    circuit = get_circuit(index)
    qc = circuit_to_qiskit(circuit)
    pattern = transpile(circuit)
    sv1 = pattern.simulate_pattern(input_state=BasicStates.ZERO)
    sv2 = Statevector.from_instruction(qc)
    assert isinstance(sv1, Statevec)
    assert fidelity(
        sv1.psi.transpose(*reversed(range(len(sv1.psi.shape)))).flatten(), sv2.data
    ) == pytest.approx(1)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_estimate_pattern_vs_qiskit(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    index = rng.integers(ncircuits)
    circuit = get_circuit(index)
    qc = circuit_to_qiskit(circuit)
    pattern = transpile(circuit)
    p1 = estimate_circuit_by_expectation_value(qc)
    backend = StatevectorBackend()
    pattern.simulate_pattern(backend=backend, input_state=BasicStates.ZERO)
    p2 = 1 - backend.estimate(pattern.output_nodes[0], Measurement(0, Plane.YZ))
    assert math.isclose(p1, p2, abs_tol=1e-8)


# @pytest.mark.parametrize("jumps", range(1, 11))
# def test_estimate_circuit_vs_pattern_sampling(fx_bg: PCG64, jumps: int) -> None:
#    rng = Generator(fx_bg.jumped(jumps))
#    index = rng.integers(ncircuits)
#    circuit = get_circuit(index)
#    qiskit_circuit = circuit_to_qiskit(circuit)
#    p1 = estimate_circuit_by_expectation_value(qiskit_circuit)
#    pattern = transpile(circuit)
#    output_node = pattern.output_nodes[0]
#    pattern.add(command.M(output_node))
#    #pattern.perform_pauli_measurements()
#    #pattern.minimize_space()
#    nb_shots = 2 << 8
#    nb_one_outcomes = 0
#    for _ in range(nb_shots):
#        measure_method = DefaultMeasureMethod()
#        pattern.simulate_pattern(measure_method=measure_method)
#        if measure_method.get_measure_result(output_node):
#            nb_one_outcomes += 1
#    p2 = nb_one_outcomes / nb_shots
#    assert math.isclose(p1, p2, abs_tol=1e-2)
