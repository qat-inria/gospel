from pathlib import Path
import numpy as np
import qiskit
import qiskit.qasm2
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import BackendEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator


def sample_circuit(nqubits, depth, p_gate, p_cnot, rng):
    qc = QuantumCircuit(QuantumRegister(nqubits), ClassicalRegister(1))
    for _ in range(depth):
        qubit = 0
        while qubit < nqubits:
            if rng.random() < p_gate:
                if qubit < nqubits - 1 and rng.random() < p_cnot:
                    qc.cx(qubit, qubit + 1)
                    qubit += 2
                else:
                    qc.rx(rng.random() * 2 * np.pi, qubit)
                    qubit += 1
            else:
                qubit += 1
    qc.measure(0, 0)
    return qc


def estimate_circuit(qc):
    pauli_string = "Z" + "I" * (qc.num_qubits - 1)
    observable = SparsePauliOp(pauli_string)
    simulator = AerSimulator(method="statevector")
    estimator = BackendEstimatorV2(backend=simulator)
    job = estimator.run([(qc, observable)])
    exp_val = job.result()[0].data.evs
    return (1 - exp_val) / 2


def generate_circuits(ncircuits, nqubits, depth, p_gate, p_cnot, rng):
    circuits = [
        sample_circuit(nqubits, depth, p_gate, p_cnot, rng) for _ in range(ncircuits)
    ]
    return circuits


def save_circuits(circuits):
    for i, circuit in enumerate(circuits):
        with Path(f"circuits/circuit{i:04}.qasm").open("w") as f:
            qiskit.qasm2.dump(circuit, f)


def plot_distribution(circuits):
    samples = [estimate_circuit(circuit) for circuit in circuits]
    plt.hist(samples, bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Random Samples (0 to 1)")
    plt.savefig("distribution.svg")


rng = np.random.default_rng()
circuits = generate_circuits(10000, 5, 10, 0.5, 0.5, rng)
save_circuits(circuits)
plot_distribution(circuits)
