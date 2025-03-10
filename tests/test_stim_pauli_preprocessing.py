import numpy as np
import numpy.typing as npt
import pytest
import stim
from graphix import Circuit, Pattern, command
from graphix.fundamentals import Plane
from graphix.noise_models.depolarising_noise_model import DepolarisingNoiseModel
from graphix.noise_models.noise_model import NoiseModel
from graphix.random_objects import rand_circuit
from graphix.sim.base_backend import (
    FixedBranchSelector,
    RandomBranchSelector,
)
from graphix.simulator import DefaultMeasureMethod
from numpy.random import PCG64, Generator

from gospel.scripts import compare_backend_results
from gospel.stim_pauli_preprocessing import (
    StimBackend,
    cut_pattern,
    preprocess_pauli,
    simulate_pauli,
)


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
def test_pauli_measurement_random_circuit(
    fx_bg: PCG64, jumps: int, backend: str
) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 4
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals()
    pattern2 = preprocess_pauli(pattern, leave_input=False)
    pattern.minimize_space()
    pattern2.minimize_space()
    # Since the patterns are deterministic, we do not need to select a particular branch
    state = pattern.simulate_pattern(backend)
    state2 = pattern2.simulate_pattern(backend)
    assert compare_backend_results(state, state2) == pytest.approx(1)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_branch_selection(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 4
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals()
    pattern_a = preprocess_pauli(pattern, leave_input=False)
    pattern_b = preprocess_pauli(pattern, leave_input=False, branch=pattern_a.results)
    assert list(pattern_a) == list(pattern_b)


@pytest.mark.parametrize("jumps", range(1, 2))
def test_simulate_pauli(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 4
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals()
    pattern.move_pauli_measurements_to_the_front()
    pattern2 = preprocess_pauli(pattern, leave_input=False)
    sim = stim.TableauSimulator()
    pauli_pattern, non_pauli_pattern = cut_pattern(pattern)
    backend = StimBackend(sim, branch=pattern2.results)
    measure_method = DefaultMeasureMethod()
    pauli_pattern.simulate_pattern(backend, measure_method=measure_method)
    output_node_set = set(pauli_pattern.output_nodes)
    input_nodes = [node for node in pattern.input_nodes if node in output_node_set]
    second_pattern = backend.to_pattern(input_nodes, non_pauli_pattern.input_nodes)
    second_pattern.extend(non_pauli_pattern)
    pattern.minimize_space()
    second_pattern.standardize()
    second_pattern.results = measure_method.results
    second_pattern.minimize_space()
    state = pattern.simulate_pattern()
    state2 = second_pattern.simulate_pattern()
    assert compare_backend_results(state, state2) == pytest.approx(1)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_simulate_pauli_depolarising_noise(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 4
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals()
    pattern.move_pauli_measurements_to_the_front()
    pauli_pattern, _non_pauli_pattern = cut_pattern(pattern)
    sim = stim.TableauSimulator()
    noise_model = DepolarisingNoiseModel()
    simulate_pauli(sim, pauli_pattern, noise_model)


def hpat() -> Pattern:
    circ = Circuit(1)
    circ.h(0)
    return circ.transpile().pattern


def simulate_with_noise_model_to_density_matrix(
    pattern: Pattern, noise_model: NoiseModel
) -> npt.NDArray[np.complex128]:
    backend = StimBackend()
    pattern.simulate_pattern(backend=backend, noise_model=noise_model)
    second_pattern = backend.to_pattern([], pattern.output_nodes)
    state = second_pattern.simulate_pattern()
    return np.outer(state.psi, state.psi.conj())


def test_noisy_measure_confuse_hadamard() -> None:
    hadamard_pattern = hpat()
    noise_model = DepolarisingNoiseModel(measure_error_prob=1.0)
    rho = simulate_with_noise_model_to_density_matrix(hadamard_pattern, noise_model)
    # result should be |1>
    assert np.allclose(rho, np.array([[0.0, 0.0], [0.0, 1.0]]))


@pytest.mark.parametrize("jumps", range(1, 11))
def test_noisy_measure_confuse_hadamard_random(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    hadamard_pattern = hpat()
    noise_model = DepolarisingNoiseModel(measure_error_prob=rng.random())
    rho = simulate_with_noise_model_to_density_matrix(hadamard_pattern, noise_model)
    assert np.allclose(rho, np.array([[1.0, 0.0], [0.0, 0.0]])) or np.allclose(
        rho,
        np.array([[0.0, 0.0], [0.0, 1.0]]),
    )
