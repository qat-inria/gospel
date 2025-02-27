import math

import numpy as np
import pytest
from graphix import Circuit
from graphix.sim.statevec import Statevec
from numpy.random import PCG64, Generator

from gospel.brickwork_state_transpiler import (
    CNOT,
    Layer,
    SingleQubit,
    SingleQubitPair,
    get_bipartite_coloring,
    transpile,
    transpile_to_layers,
)
from gospel.sampling_circuits import get_circuit, ncircuits


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
                SingleQubitPair(SingleQubit(), SingleQubit()),
            ],
        ),
        Layer(True, [CNOT(target_above=False)]),
        Layer(
            False,
            [
                CNOT(target_above=True),
                SingleQubitPair(SingleQubit(rx=math.pi / 4), SingleQubit()),
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


def check_circuit(circuit: Circuit) -> None:
    pattern = transpile(circuit)
    sv1 = circuit.simulate_statevector().statevec
    sv2 = pattern.simulate_pattern()
    assert isinstance(sv2, Statevec)
    assert np.abs(np.dot(sv1.flatten().conjugate(), sv2.flatten())) == pytest.approx(1)


def test_transpile_rz() -> None:
    circuit = Circuit(2)
    circuit.rz(0, 0.1)
    check_circuit(circuit)


def test_transpile_rx() -> None:
    circuit = Circuit(2)
    circuit.rx(0, 0.1)
    check_circuit(circuit)


def test_transpile_cnot() -> None:
    circuit = Circuit(2)
    circuit.cnot(0, 1)
    check_circuit(circuit)


def test_transpile_rz_rx_cnot() -> None:
    circuit = Circuit(2)
    circuit.rz(0, 0.1)
    circuit.rz(1, 0.1)
    circuit.cnot(0, 1)
    check_circuit(circuit)


def test_transpile_multiple_cnot() -> None:
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
    check_circuit(circuit)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_sampled_circuit(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    circuit = get_circuit(rng.integers(ncircuits))
    check_circuit(circuit)


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
