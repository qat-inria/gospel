from __future__ import annotations

import json
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import qiskit
import qiskit.qasm2
import typer
from graphix import Circuit
from graphix.instruction import InstructionKind
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli, Statevector
from tqdm import tqdm

if TYPE_CHECKING:
    from pathlib import Path


def sample_circuit(
    nqubits: int,
    depth: int,
    p_gate: float,
    p_cnot: float,
    p_cnot_flip: float,
    p_rx: float,
    rng: np.random.Generator,
) -> Circuit:
    """
    Generates a quantum circuit with the given number of qubits and depth.

    At each layer (depth), the circuit iterates through the qubits. With probability
    `p_gate`, a gate is applied. If there is room (i.e. not on the last qubit) and
    with probability `p_cnot`, a controlled-NOT (CX) gate is applied between the current
    qubit and the next one (skipping the next qubit): with probability `p_cnot_flip`,
    the target is the current qubit and the control the next one, otherwise the converse.
    Otherwise, a rotation gate with a random angle is applied to the current qubit:
    with probability `p_rx`, the rotation is RX, otherwise RZ.

    The circuit is then stripped so that gates are kept only if they can affect qubit 0.
    Rotations on other qubits that are not followed with a CNOT connecting them to
    qubit 0 are removed.
    """
    circuit = Circuit(nqubits)
    last_operation = {}
    for _ in range(depth):
        qubit = 0
        while qubit < nqubits:
            if rng.random() < p_gate:
                last = last_operation.get(qubit)
                # Check if there's room for a CX gate and with probability p_cnot, apply CX.
                if qubit < nqubits - 1 and rng.random() < p_cnot:
                    last_next = last_operation.get(qubit + 1, None)
                    if (last, last_next) == ("control", "target") or (
                        (last, last_next) != ("target", "control")
                        and rng.random() < p_cnot_flip
                    ):
                        control = qubit + 1
                        target = qubit
                    else:
                        control = qubit
                        target = qubit + 1
                    circuit.cnot(control, target)
                    last_operation[control] = "control"
                    last_operation[target] = "target"
                    qubit += 2  # Skip the next qubit since it's already involved in CX
                else:
                    angle = rng.random() * 2 * np.pi
                    # With probability p_rx, apply RX; otherwise, apply RZ.
                    if last == "rz" or (last != "rx" and rng.random() < p_rx):
                        circuit.rx(qubit, angle)
                        last_operation[qubit] = "rx"
                    else:
                        circuit.rz(qubit, angle)
                        last_operation[qubit] = "rz"
                    qubit += 1
            else:
                # No gate applied; move to the next qubit.
                qubit += 1
    # Initialize an empty list for the instructions that remain after stripping.
    new_instructions = []
    # 'reachable' tracks the index of the last qubit that can affect qubit 0.
    reachable = 0
    for instr in reversed(circuit.instruction):
        match instr.kind:
            case InstructionKind.CNOT:
                # By construction, instr.target == instr.control + 1.
                # If the control qubit is beyond the current reachable range,
                # the gate cannot affect qubit 0 and is removed.
                if instr.control > reachable:
                    continue
                # If the control qubit is exactly at the reachable boundary,
                # this CX gate extends the influence to the next qubit.
                if instr.control == reachable:
                    reachable += 1
                # Keep the instruction.
                new_instructions.append(instr)
            case InstructionKind.RX | InstructionKind.RZ:
                # Keep the rotation only if it can affect a qubit within the reachable range.
                if instr.target <= reachable:
                    new_instructions.append(instr)
    # The instructions were collected in reverse order; reverse them to restore original order.
    new_instructions.reverse()
    # Replace the original instruction list with the new, stripped list.
    circuit.instruction = new_instructions
    return circuit


def circuit_to_qiskit(c: Circuit) -> QuantumCircuit:
    """
    Convert a Graphix circuit to a Qiskit QuantumCircuit.

    Parameters:
        c (Circuit): Graphix circuit

    Returns:
        QuantumCircuit: A Qiskit QuantumCircuit representing the custom circuit.

    Raises:
        ValueError: If an instruction type is not supported.
    """
    qc = QuantumCircuit(QuantumRegister(c.width), ClassicalRegister(1))
    for instr in c.instruction:
        match instr.kind:
            case InstructionKind.CNOT:
                # Qiskit's cx method expects (control, target).
                qc.cx(instr.control, instr.target)
            case InstructionKind.RX:
                qc.rx(instr.angle, instr.target)
            case InstructionKind.RZ:
                qc.rz(instr.angle, instr.target)
            case _:
                raise ValueError(f"Unsupported instruction: {instr.kind}")
    return qc


## Alternative method for estimating probability by sampling
# def estimate_circuit(qc: QuantumCircuit, seed: int | None = None) -> float:
#    """
#    Estimate the probability of measuring the '1' outcome on the first qubit.
#    """
#    qc.measure(0, 0)
#    nb_shots = 2 << 8
#    sampler = SamplerV2(seed=seed)
#    job = sampler.run([qc], shots=nb_shots)
#    job_result = job.result()
#    return sum(next(iter(job_result[0].data.values())).bitcount()) / nb_shots


def estimate_circuit_expectation_value(qc: QuantumCircuit) -> float:
    """
    Estimate the probability of measuring the '1' outcome on the first qubit.

    The observable is chosen as Z on the first qubit (and I on all others),
    so that the expectation value <Z> is computed on the first qubit.
    Given that for a qubit in state |ψ⟩:
        <Z> = p(0) - p(1)
    the probability of outcome '1' is computed as:
        p(1) = (1 - <Z>) / 2
    """
    # Get the statevector for the circuit
    sv = Statevector.from_instruction(qc)
    # Compute the expectation value of the observable
    exp_val = sv.expectation_value(Pauli("Z"), [0])
    assert np.imag(exp_val) == 0
    # p(1) = (1 - <Z>)/2
    return (1 - np.real(exp_val)) / 2


def generate_circuits(
    ncircuits: int,
    nqubits: int,
    depth: int,
    p_gate: float,
    p_cnot: float,
    p_cnot_flip: float,
    p_rx: float,
    rng: np.random.Generator,
) -> list[Circuit]:
    """
    Generate a list of quantum circuits with the given number of qubits and depth.
    """
    return [
        sample_circuit(nqubits, depth, p_gate, p_cnot, p_cnot_flip, p_rx, rng)
        for _ in range(ncircuits)
    ]


def estimate_circuits(
    circuits: list[QuantumCircuit],
) -> list[tuple[QuantumCircuit, float]]:
    return [
        (circuit, estimate_circuit_expectation_value(circuit))
        for circuit in tqdm(circuits)
    ]


def save_circuits(
    circuits: list[tuple[QuantumCircuit, float]], threshold: float, path: Path
) -> None:
    table = {}
    maxlen = int(np.log10(len(circuits) - 1) + 1)
    for i, (circuit, p) in enumerate(circuits):
        filename = f"circuit{str(i).zfill(maxlen)}.qasm"
        with (path / filename).open("w") as f:
            qiskit.qasm2.dump(circuit, f)
        table[filename] = p
    with (path / "table.json").open("w") as f:
        json.dump(table, f)


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
    p_cnot_flip: float = typer.Option(..., help="Probability of flipping a CNOT gate"),
    p_rx: float = typer.Option(..., help="Probability of applying an RX gate"),
    seed: int = typer.Option(..., help="Random seed"),
    threshold: float = typer.Option(..., help="Threshold value"),
    target: Path = typer.Option(..., help="Target directory"),
) -> None:
    params = locals()
    rng = np.random.default_rng(seed=seed)
    target.mkdir()
    circuits = generate_circuits(
        ncircuits, nqubits, depth, p_gate, p_cnot, p_cnot_flip, p_rx, rng
    )
    qiskit_circuits = map(circuit_to_qiskit, circuits)
    estimated_circuits = estimate_circuits(qiskit_circuits)
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
