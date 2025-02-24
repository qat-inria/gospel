import math

import numpy as np
import pytest
from graphix import Circuit
from qiskit import QuantumCircuit
from qiskit.quantum_info import process_fidelity


def test_cnot_graphix():
    circuit = Circuit(2)
    circuit.rz(0, math.pi / 2)
    circuit.rx(1, math.pi / 2)
    circuit.cz(0, 1)
    circuit.rx(1, -math.pi / 2)
    circuit.cz(0, 1)
    circuit2 = Circuit(2)
    circuit2.cnot(0, 1)
    sv1 = circuit.simulate_statevector().statevec
    sv2 = circuit2.simulate_statevector().statevec
    assert np.abs(np.dot(sv1.flatten().conjugate(), sv2.flatten())) == pytest.approx(1)


def test_rz_cnot_graphix():
    circuit = Circuit(2)
    circuit.rz(1, math.pi / 4)
    circuit.rz(0, math.pi / 2)
    circuit.rx(1, math.pi / 2)
    circuit.cz(0, 1)
    circuit.rx(1, -math.pi / 2)
    circuit.cz(0, 1)
    circuit2 = Circuit(2)
    circuit2.rz(1, math.pi / 4)
    circuit2.cnot(0, 1)
    sv1 = circuit.simulate_statevector().statevec
    sv2 = circuit2.simulate_statevector().statevec
    assert np.abs(np.dot(sv1.flatten().conjugate(), sv2.flatten())) == pytest.approx(1)


def test_cnot_qiskit():
    circuit = QuantumCircuit(2)
    # First block of rotations
    circuit.rx(math.pi / 2, 1)
    # First entangling gate
    circuit.cz(0, 1)
    # Second block of rotations
    circuit.rz(math.pi / 2, 0)
    circuit.rx(-math.pi / 2, 1)
    circuit.cz(0, 1)
    circuit2 = QuantumCircuit(2)
    circuit2.cx(0, 1)
    from qiskit.quantum_info import Operator

    op = Operator.from_circuit(circuit)
    op2 = Operator.from_circuit(circuit2)
    assert process_fidelity(op, op2) == pytest.approx(1)
