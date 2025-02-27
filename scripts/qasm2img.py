from pathlib import Path

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from tqdm import tqdm


def qasm2img(source: Path, target: Path) -> None:
    qc = QuantumCircuit.from_qasm_file(str(source))
    qc.measure(0, 0)
    qc.draw(output="mpl", filename=str(target))
    plt.close()


def convert_circuit_directory() -> None:
    circuits_path = Path("circuits")
    circuits_svg_path = Path("circuits_svg")
    circuits_svg_path.mkdir()
    for circuit in tqdm(circuits_path.glob("*.qasm")):
        qasm2img(circuit, (circuits_svg_path / circuit.name).with_suffix(".svg"))


if __name__ == "__main__":
    convert_circuit_directory()
