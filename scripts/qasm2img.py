from pathlib import Path

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit


def qasm2img(source: Path, target: Path) -> None:
    qc = QuantumCircuit.from_qasm_file(source)
    qc.draw(output="mpl", filename=target)
    plt.close()


def convert_circuit_directory():
    circuits_path = Path("circuits")
    circuits_svg_path = Path("circuits_svg")
    circuits_svg_path.mkdir()
    for circuit in circuits_path.glob("*.qasm"):
        qasm2img(circuit, (circuits_svg_path / circuit.name).with_suffix(".svg"))


if __name__ == "__main__":
    convert_circuit_directory()
