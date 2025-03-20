import math

import pytest
import stim
from graphix import Circuit, Pattern
from graphix.command import CommandKind
from numpy.random import PCG64, Generator

from gospel.brickwork_state_transpiler import (
    CNOT,
    ConstructionOrder,
    Layer,
    SingleQubit,
    SingleQubitPair,
    generate_random_pauli_pattern,
    get_bipartite_coloring,
    get_hot_traps_of_faulty_gate,
    get_node_positions,
    layers_to_circuit,
    transpile,
    transpile_to_layers,
)
from gospel.sampling_circuits import get_circuit, ncircuits, sample_circuit
from gospel.scripts import compare_backend_results
from gospel.stim_pauli_preprocessing import simulate_pauli


def test_transpile_to_layers_rx_rz_on_two_qubits() -> None:
    circuit = Circuit(2)
    circuit.rx(0, math.pi / 4)
    circuit.rz(1, math.pi / 4)
    layers = transpile_to_layers(circuit)
    assert layers == [
        Layer(
            False,
            [
                SingleQubitPair(
                    SingleQubit(rx=math.pi / 4), SingleQubit(rz0=math.pi / 4)
                )
            ],
        )
    ]


def test_transpile_to_layers_rx_rz_on_the_same_qubit() -> None:
    circuit = Circuit(1)
    circuit.rx(0, math.pi / 4)
    circuit.rz(0, math.pi / 4)
    layers = transpile_to_layers(circuit)
    assert layers == [
        Layer(
            False,
            [
                SingleQubitPair(
                    SingleQubit(rx=math.pi / 4, rz1=math.pi / 4), SingleQubit()
                )
            ],
        )
    ]


def test_transpile_to_layers_cnot_rx_cnot_rx() -> None:
    circuit = Circuit(3)
    circuit.cnot(1, 2)
    circuit.rx(0, math.pi / 4)
    circuit.cnot(1, 0)
    circuit.rx(2, math.pi / 4)
    layers = transpile_to_layers(circuit)
    assert layers == [
        Layer(
            False,
            [
                SingleQubitPair(SingleQubit(rx=math.pi / 4), SingleQubit()),
            ],
        ),
        Layer(True, [CNOT(target_above=False)]),
        Layer(
            False,
            [
                CNOT(target_above=True),
            ],
        ),
        Layer(
            True,
            [
                SingleQubitPair(SingleQubit(), SingleQubit(rx=math.pi / 4)),
            ],
        ),
    ]


def test_transpile_to_layers_rx_rz_rx() -> None:
    circuit = Circuit(1)
    circuit.rx(0, math.pi / 4)
    circuit.rz(0, math.pi / 4)
    circuit.rx(0, math.pi / 4)
    layers = transpile_to_layers(circuit)
    assert layers == [
        Layer(
            False,
            [
                SingleQubitPair(
                    SingleQubit(rx=math.pi / 4, rz1=math.pi / 4), SingleQubit()
                )
            ],
        ),
        Layer(True, []),
        Layer(False, [SingleQubitPair(SingleQubit(rx=math.pi / 4), SingleQubit())]),
    ]


def test_transpile_to_layers_four_cnots() -> None:
    circuit = Circuit(4)
    circuit.cnot(0, 1)
    circuit.cnot(1, 2)
    circuit.rx(1, math.pi / 4)
    circuit.rx(2, math.pi / 4)
    circuit.rx(3, math.pi / 4)
    circuit.cnot(2, 3)
    circuit.rx(0, math.pi / 4)
    circuit.cnot(0, 1)
    layers = transpile_to_layers(circuit)
    assert layers == [
        Layer(
            False,
            [
                CNOT(target_above=False),
                SingleQubitPair(SingleQubit(), SingleQubit(rx=math.pi / 4)),
            ],
        ),
        Layer(True, [CNOT(target_above=False)]),
        Layer(
            False,
            [
                SingleQubitPair(
                    SingleQubit(rx=math.pi / 4), SingleQubit(rx=math.pi / 4)
                ),
                SingleQubitPair(SingleQubit(rx=math.pi / 4), SingleQubit()),
            ],
        ),
        Layer(True, [SingleQubitPair(SingleQubit(), SingleQubit())]),
        Layer(False, [CNOT(target_above=False), CNOT(target_above=False)]),
    ]


def check_order(pattern: Pattern, order: ConstructionOrder) -> None:
    width = len(pattern.input_nodes)
    positions = get_node_positions(pattern)
    has_horizontal_bar = set()
    for cmd in pattern:
        if cmd.kind == CommandKind.E:
            u, v = cmd.nodes
            ux, uy = positions[u]
            vx, vy = positions[v]
            if ux == vx:
                if ux < pattern.n_node // width - 1:
                    if order == ConstructionOrder.DeviantRight:
                        assert (ux, uy) in has_horizontal_bar
                        assert (vx, vy) in has_horizontal_bar
                    else:
                        assert (ux, uy) not in has_horizontal_bar
                        assert (vx, vy) not in has_horizontal_bar
            else:
                assert uy == vy
                if order == ConstructionOrder.Deviant:
                    has_horizontal_bar.add((max(ux, vx), uy))
                else:
                    has_horizontal_bar.add((min(ux, vx), uy))


def check_circuit(circuit: Circuit, order: ConstructionOrder) -> None:
    pattern = transpile(circuit, order)
    check_order(pattern, order)
    sv1 = circuit.simulate_statevector().statevec
    sv2 = pattern.simulate_pattern()
    assert compare_backend_results(sv1, sv2) == pytest.approx(1)


@pytest.mark.parametrize("order", list(ConstructionOrder))
def test_transpile_rz(order: ConstructionOrder) -> None:
    circuit = Circuit(2)
    circuit.rz(0, 0.1)
    check_circuit(circuit, order)


@pytest.mark.parametrize("order", list(ConstructionOrder))
def test_transpile_rx(order: ConstructionOrder) -> None:
    circuit = Circuit(2)
    circuit.rx(0, 0.1)
    check_circuit(circuit, order)


@pytest.mark.parametrize("order", list(ConstructionOrder))
def test_transpile_cnot(order: ConstructionOrder) -> None:
    circuit = Circuit(2)
    circuit.cnot(0, 1)
    check_circuit(circuit, order)


@pytest.mark.parametrize("order", list(ConstructionOrder))
def test_transpile_rz_rx_cnot(order: ConstructionOrder) -> None:
    circuit = Circuit(2)
    circuit.rz(0, 0.1)
    circuit.rz(1, 0.1)
    circuit.cnot(0, 1)
    check_circuit(circuit, order)


@pytest.mark.parametrize("order", list(ConstructionOrder))
def test_transpile_multiple_cnot(order: ConstructionOrder) -> None:
    circuit = Circuit(4)
    circuit.rz(0, 0.1)
    circuit.rx(0, 0.1)
    circuit.rx(1, 0.1)
    circuit.rz(1, 0.1)
    circuit.rx(2, 0.1)
    circuit.rz(3, 0.1)
    circuit.cnot(0, 1)
    circuit.cnot(2, 3)
    circuit.cnot(1, 2)
    check_circuit(circuit, order)


@pytest.mark.parametrize("jumps", range(1, 11))
@pytest.mark.parametrize("order", list(ConstructionOrder))
def test_sampled_circuit(fx_bg: PCG64, jumps: int, order: ConstructionOrder) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    circuit = get_circuit(rng.integers(ncircuits))
    check_circuit(circuit, order)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_get_bipartite_coloring(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    circuit = get_circuit(rng.integers(ncircuits))
    pattern = transpile(circuit)
    red, blue = get_bipartite_coloring(pattern)
    nodes, edges = pattern.get_graph()
    assert red | blue == set(nodes)
    assert red & blue == set()
    for edge in edges:
        assert (edge[0] in red and edge[1] in blue) or (
            edge[0] in blue and edge[1] in red
        )


@pytest.mark.parametrize("jumps", range(1, 11))
@pytest.mark.parametrize("order", list(ConstructionOrder))
def test_generate_random_pauli_pattern(
    fx_bg: PCG64, jumps: int, order: ConstructionOrder
) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    pattern = generate_random_pauli_pattern(nqubits=8, nlayers=10, rng=rng, order=order)
    sim = stim.TableauSimulator()
    simulate_pauli(sim, pattern)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_layers_to_circuit(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    circuit = sample_circuit(nqubits=8, depth=10, rng=rng)
    layers = transpile_to_layers(circuit)
    circuit2 = layers_to_circuit(layers)
    sv1 = circuit.simulate_statevector().statevec
    sv2 = circuit2.simulate_statevector().statevec
    assert compare_backend_results(sv1, sv2) == pytest.approx(1)


def test_get_hot_traps_of_faulty_gate() -> None:
    nqubits = 7
    # Map between a gate and the "kind" of hot traps
    #
    # *+*-*-o-o
    #     |   |
    # o-o-o-o-o-o-*+*-*
    #             |   |
    # o-*+*-*-o-o-o-o-o
    #     |   |
    # o-o-*-o-o-o-o-o-o
    #             |   |
    # o-o-*-*-o-*+*-*-o
    #     +   |
    # o-o-*-*-o-o-o-o-o
    #             |   |
    #         o-o-*+*-*
    faulty_gates = {
        (0, 7): (0, {0, 7, 14}),
        (9, 16): (1, {9, 16, 17, 23}),
        (18, 19): (2, {18, 19, 25, 26}),
        (43, 50): (3, {43, 50, 57}),
        (39, 46): (4, {39, 46, 53}),
        (48, 55): (5, {48, 55, 62}),
        (1, 8): (None, set()),
    }
    for faulty_gate, expected_answer in faulty_gates.items():
        assert get_hot_traps_of_faulty_gate(nqubits, faulty_gate) == expected_answer
