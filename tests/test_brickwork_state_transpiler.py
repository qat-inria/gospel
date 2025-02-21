import math

import numpy as np
import pytest
from graphix import Circuit

from brickwork_state_transpiler import (
    CNot,
    Layer,
    SingleQubit,
    SingleQubitPair,
    transpile,
    transpile_to_layers,
)


def test_transpile_to_layers_rx_rz_on_two_qubits():
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


def test_transpile_to_layers_rx_rz_on_the_same_qubit():
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


def test_transpile_to_layers_cnot_rx_cnot_rx():
    circuit = Circuit(3)
    circuit.cnot(1, 2)
    circuit.rx(0, math.pi / 4)
    circuit.cnot(0, 1)
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
        Layer(True, [CNot]),
        Layer(
            False, [CNot, SingleQubitPair(SingleQubit(rx=math.pi / 4), SingleQubit())]
        ),
    ]


def test_transpile_to_layers_rx_rz_rx():
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


def test_transpile_to_layers_four_cnots():
    circuit = Circuit(4)
    circuit.cnot(0, 1)
    circuit.cnot(1, 2)
    circuit.rx(1, math.pi / 4)
    circuit.rx(3, math.pi / 4)
    circuit.rx(2, math.pi / 4)
    circuit.cnot(2, 3)
    circuit.rx(0, math.pi / 4)
    circuit.cnot(0, 1)
    layers = transpile_to_layers(circuit)
    assert layers == [
        Layer(
            False, [CNot, SingleQubitPair(SingleQubit(), SingleQubit(rx=math.pi / 4))]
        ),
        Layer(True, [CNot]),
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
        Layer(False, [CNot, CNot]),
    ]


def test_transpile_cnot():
    circuit = Circuit(2)
    # circuit.rx(0, math.pi / 4)
    circuit.rz(0, math.pi / 4)
    # circuit.cnot(0, 1)
    pattern = transpile(circuit)
    print(list(pattern))
    sv1 = circuit.simulate_statevector().statevec
    sv2 = pattern.simulate_pattern()
    assert np.abs(np.dot(sv1.flatten().conjugate(), sv2.flatten())) == pytest.approx(1)
