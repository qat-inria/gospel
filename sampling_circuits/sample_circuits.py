from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qiskit
import qiskit.qasm2
import typer
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.primitives import BackendEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import StatevectorSimulator
from tqdm import tqdm


def sample_circuit(
    nqubits: int, depth: int, p_gate: float, p_cnot: float, rng: np.random.Generator
) -> QuantumCircuit:
    """
    Generates a quantum circuit with the given number of qubits and depth.

    At each layer (depth), the circuit iterates through the qubits. With probability
    p_gate, a gate is applied. If there is room (i.e. not on the last qubit) and
    with probability p_cnot, a controlled-NOT (CX) gate is applied between the current
    qubit and the next one (skipping the next qubit). Otherwise, a rotation gate (rx)
    with a random angle is applied to the current qubit.

    Finally, qubit 0 is measured into the single classical bit.
    """
    qc = QuantumCircuit(QuantumRegister(nqubits), ClassicalRegister(1))
    for _ in range(depth):
        qubit = 0
        while qubit < nqubits:
            if rng.random() < p_gate:
                # If there's room for a CX gate and with probability p_cnot, apply CX
                if qubit < nqubits - 1 and rng.random() < p_cnot:
                    qc.cx(qubit, qubit + 1)
                    qubit += 2  # Skip the next qubit since it's already involved in CX
                else:
                    qc.rx(rng.random() * 2 * np.pi, qubit)
                    qubit += 1
            else:
                qubit += 1
    # Measure the first qubit (qubit 0) into the classical register
    qc.measure(0, 0)
    return qc


def estimate_circuit(qc: QuantumCircuit) -> float:
    """
    Estimate the probability of measuring the '1' outcome on the first qubit.

    The observable is chosen as Z on the first qubit (and I on all others),
    so that the expectation value <Z> is computed on the first qubit.
    Given that for a qubit in state |ψ⟩:
        <Z> = p(0) - p(1)
    the probability of outcome '1' is computed as:
        p(1) = (1 - <Z>) / 2

    Note:
        If the circuit qc contains an explicit measurement on the first qubit,
        that measurement gate is typically removed (or ignored) by the estimator
        when using a statevector simulator. That is, the estimator computes the
        expectation value on the pre-measurement state, so the measurement does not
        “trace out” the qubit in the simulation.
    """
    # Create an observable that acts as Z on the first qubit and Identity on the rest.
    pauli_string = "Z" + "I" * (qc.num_qubits - 1)
    observable = SparsePauliOp(pauli_string)
    simulator = StatevectorSimulator()
    estimator = BackendEstimatorV2(backend=simulator)
    # Run the estimator: note that measurement operations in qc (if any) are typically ignored
    # so that the expectation value is computed on the unitary (pre-measurement) part.
    job = estimator.run([(qc, observable)])
    exp_val = job.result()[0].data.evs
    # Compute the probability of outcome "1" (eigenvalue -1 of Z) using:
    # p(1) = (1 - <Z>)/2.
    return (1 - exp_val) / 2


def generate_circuits(
    ncircuits: int,
    nqubits: int,
    depth: int,
    p_gate: float,
    p_cnot: float,
    rng: np.random.Generator,
) -> list[QuantumCircuit]:
    """
    Generate a list of quantum circuits with the given number of qubits and depth.
    """
    return [
        sample_circuit(nqubits, depth, p_gate, p_cnot, rng) for _ in range(ncircuits)
    ]


def estimate_circuits(
    circuits: list[QuantumCircuit],
) -> list[tuple[QuantumCircuit, float]]:
    return [(circuit, estimate_circuit(circuit)) for circuit in tqdm(circuits)]


def save_circuits(circuits: list[QuantumCircuit], threshold: float, root: Path) -> None:
    no_path = root / f"0-{threshold}"
    other_path = root / f"{threshold}-{1 - threshold}"
    yes_path = root / f"{1 - threshold}-1"
    no_path.mkdir()
    other_path.mkdir()
    yes_path.mkdir()
    maxlen = int(np.log10(len(circuits) - 1) + 1)
    for i, (circuit, p) in enumerate(circuits):
        if p <= threshold:
            path = no_path
        elif p >= 1 - threshold:
            path = yes_path
        else:
            path = other_path
        with (path / f"circuit{str(i).zfill(maxlen)}.qasm").open("w") as f:
            qiskit.qasm2.dump(circuit, f)


def plot_distribution(
    circuits: list[tuple[QuantumCircuit, float]], filename: Path
) -> None:
    samples = [p for _circuit, p in circuits]
    plt.hist(samples, bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Random Samples (0 to 1)")
    plt.savefig(filename)


def sample_circuits(
    ncircuits: int = typer.Option(..., help="Number of circuits"),
    nqubits: int = typer.Option(..., help="Number of qubits"),
    depth: int = typer.Option(..., help="Circuit depth"),
    p_gate: float = typer.Option(..., help="Probability of applying a gate"),
    p_cnot: float = typer.Option(..., help="Probability of applying a CNOT gate"),
    seed: int = typer.Option(..., help="Random seed"),
    threshold: float = typer.Option(..., help="Threshold value"),
    target: Path = typer.Option(..., help="Target directory"),
) -> None:
    params = locals()
    rng = np.random.default_rng(seed=seed)
    target.mkdir()
    circuits = generate_circuits(ncircuits, nqubits, depth, p_gate, p_cnot, rng)
    estimated_circuits = estimate_circuits(circuits)
    save_circuits(estimated_circuits, threshold, target)
    plot_distribution(estimated_circuits, target / "distribution.svg")
    arg_str = " ".join(
        f"--{key.replace('_', '-')} {value}" for key, value in params.items()
    )
    command_line = f"python sample_circuits.py {arg_str}"
    with (target / "README.md").open("w") as f:
        f.write(f"""To reproduce these samples, you may run the following command:
```
{command_line}
```
""")


if __name__ == "__main__":
    typer.run(sample_circuits)
