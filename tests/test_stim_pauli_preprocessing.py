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
from graphix.sim.statevec import Statevec
from graphix.simulator import DefaultMeasureMethod
from graphix.states import BasicState, BasicStates
from numpy.random import PCG64, Generator
from veriphix.client import Client, Secrets
from veriphix.trappifiedCanvas import TrappifiedCanvas

from gospel.brickwork_state_transpiler import (
    ConstructionOrder,
    generate_random_pauli_pattern,
    get_bipartite_coloring,
)
from gospel.scripts import compare_backend_results
from gospel.stim_pauli_preprocessing import (
    StimBackend,
    cut_pattern,
    pattern_to_stim_circuit,
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
def test_pauli_measurement_random_circuit(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    backend = "statevector"
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
    assert isinstance(state, Statevec)
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


def test_add_nodes() -> None:
    states = [
        BasicStates.ZERO,
        BasicStates.ONE,
        BasicStates.PLUS,
        BasicStates.MINUS,
        BasicStates.PLUS_I,
        BasicStates.MINUS_I,
    ]
    stabs = [stim.PauliString(s) for s in ["+Z", "-Z", "+X", "-X", "+Y", "-Y"]]

    for i, state in enumerate(states):
        backend = StimBackend()
        backend.add_nodes([0], state)
        [
            stim_stab,
        ] = backend.sim.canonical_stabilizers()
        assert stim_stab == stabs[i]


@pytest.mark.parametrize("jumps", range(1, 11))
def test_simulation_test_round_simple(fx_bg: PCG64, jumps: int) -> None:
    # check all correct trap results when no noise
    # out of veriphix
    # run
    rng = Generator(fx_bg.jumped(jumps))

    circuit = rand_circuit(nqubits=3, depth=3)  # Circuit(width=1)
    # circuit.h(0)
    pattern = circuit.transpile().pattern
    pattern.minimize_space()

    for onode in pattern.output_nodes:
        pattern.add(command.M(node=onode))

    secrets = Secrets(r=False, a=False, theta=False)
    client = Client(pattern=pattern, secrets=secrets)
    # colours = gospel.brickwork_state_transpiler.get_bipartite_coloring(pattern)
    test_runs = client.create_test_runs()
    for j, i in enumerate(test_runs):
        print(j, i.traps_list)

    backend = StimBackend()
    run = TrappifiedCanvas(test_runs[rng.integers(len(test_runs))])
    # print("chosen run", run.traps_list)

    trap_outcomes = client.delegate_test_run(backend=backend, run=run)
    # print(sim.canonical_stabilizers())
    print(f"trap outcomes {trap_outcomes}")
    assert sum(trap_outcomes) == 0


@pytest.mark.parametrize("jumps", range(1, 11))
@pytest.mark.parametrize("order", list(ConstructionOrder))
def test_simulation_test_brickwork_state(
    fx_bg: PCG64, jumps: int, order: ConstructionOrder
) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    pattern = generate_random_pauli_pattern(nqubits=8, nlayers=10, order=order, rng=rng)
    for onode in pattern.output_nodes:
        pattern.add(command.M(node=onode))

    secrets = Secrets(r=False, a=False, theta=False)
    client = Client(pattern=pattern, secrets=secrets)
    colours = get_bipartite_coloring(pattern)
    test_runs = client.create_test_runs(manual_colouring=colours)

    backend = StimBackend()
    run = TrappifiedCanvas(test_runs[rng.integers(len(test_runs))], rng=rng)

    noise_model = DepolarisingNoiseModel(entanglement_error_prob=0.1)
    client.delegate_test_run(backend=backend, run=run, noise_model=noise_model)


def test_pattern_to_stim_circuit(fx_rng: Generator) -> None:
    nodes = 50
    planes = [Plane(p) for p in fx_rng.integers(low=1, high=4, size=nodes)]
    expected_results = [fx_rng.integers(2) == 1 for _ in range(nodes)]

    def get_input_state(node: int) -> BasicState:
        if planes[node] == Plane.XY:
            if expected_results[node]:
                return BasicState.MINUS
            return BasicState.PLUS
        if expected_results[node]:
            return BasicState.ONE
        return BasicState.ZERO

    pattern = Pattern(input_nodes=list(range(nodes)))
    for node in fx_rng.choice(range(nodes), size=nodes, replace=False):
        pattern.add(command.M(node, plane=planes[node], angle=0))
    circuit, measure_indices = pattern_to_stim_circuit(
        pattern,
        input_state={node: get_input_state(node) for node in range(nodes)},
    )
    sample = circuit.compile_sampler().sample(shots=1000000)
    for shot in sample:
        assert [shot[measure_indices[i]] for i in range(nodes)] == expected_results


def test_pattern_to_stim_circuit_hadamard(fx_rng: Generator) -> None:
    circuit = Circuit(2)
    circuit.h(0)
    circuit.h(1)
    pattern = circuit.transpile().pattern
    node0 = pattern.output_nodes[0]
    node1 = pattern.output_nodes[1]
    pattern.add(command.M(node0, plane=Plane.XY))
    pattern.add(command.M(node1, plane=Plane.XY))
    stim_circuit, measure_indices = pattern_to_stim_circuit(
        pattern,
        input_state={0: BasicState.ZERO, 1: BasicState.ONE},
    )
    sample = stim_circuit.compile_sampler().sample(shots=1000)
    for s in sample:
        assert not s[measure_indices[node0]]
        assert s[measure_indices[node1]]


@pytest.mark.parametrize("jumps", range(1, 11))
def test_estimate_stim_backend(fx_bg: PCG64, jumps: int) -> None:
    nshots = 1000
    rng = Generator(fx_bg.jumped(jumps))
    order = ConstructionOrder.Canonical
    pattern = generate_random_pauli_pattern(nqubits=3, nlayers=4, order=order, rng=rng)
    outcomes: dict[int, int] = {}
    for _ in range(nshots):
        backend = StimBackend()
        measure_method = DefaultMeasureMethod()
        pattern.simulate_pattern(backend, measure_method=measure_method)
        for i, v in enumerate(measure_method.results):
            if v:
                outcomes[i] = outcomes.get(i, 0) + 1
    stim_circuit, measure_indices = pattern_to_stim_circuit(pattern)
    sample = stim_circuit.compile_sampler().sample(shots=nshots)
    outcomes2: dict[int, int] = {}
    for s in sample:
        for i, v in enumerate(measure_indices):
            if s[v]:
                outcomes2[i] = outcomes2.get(i, 0) + 1
    for v, v2 in zip(outcomes, outcomes2):
        assert abs(v - v2) < 50


@pytest.mark.parametrize("jumps", range(1, 11))
def test_pattern_to_stim_circuit_round_brickwork(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    order = ConstructionOrder.Canonical
    pattern = generate_random_pauli_pattern(nqubits=8, nlayers=10, order=order, rng=rng)
    for onode in pattern.output_nodes:
        pattern.add(command.M(node=onode))
    colors = get_bipartite_coloring(pattern)
    secrets = Secrets(r=False, a=False, theta=False)
    client = Client(pattern=pattern, secrets=secrets)
    test_runs = client.create_test_runs(manual_colouring=colors)
    for col in test_runs:
        run = TrappifiedCanvas(col)
        input_state = {
            i: BasicState.try_from_statevector(Statevec(state).psi)
            for i, state in enumerate(run.states)
        }
        client_pattern = Pattern(input_nodes=pattern.input_nodes)
        for cmd in pattern:
            if isinstance(cmd, command.M):
                client_pattern.add(command.M(node=cmd.node))
            else:
                client_pattern.add(cmd)
        stim_circuit, measure_indices = pattern_to_stim_circuit(
            client_pattern,
            input_state=input_state,  # type: ignore[arg-type]
        )
        # sample = stim_circuit.compile_sampler().sample(shots=1000)
        sim = stim.TableauSimulator()
        sim.do(stim_circuit)
        sample = [sim.current_measurement_record()]
        for s in sample:
            for trap in run.traps_list:
                outcomes = [s[measure_indices[node]] for node in trap]
                trap_outcome = sum(outcomes) % 2
                assert trap_outcome == 0
