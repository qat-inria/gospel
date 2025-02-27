from pathlib import Path

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from tqdm import tqdm


def qasm2img(source: Path, target: Path) -> None:
    qc = QuantumCircuit.from_qasm_file(str(source))
    qc.measure(0, 0)
    qc.draw(output="mpl", filename=str(target))
    plt.close()


def convert_circuit_directory_to_svg(
    path_circuits: Path, path_circuits_svg: Path
) -> None:
    path_circuits_svg.mkdir()
    for circuit in tqdm(list(path_circuits.glob("*.qasm"))):
        qasm2img(circuit, (path_circuits_svg / circuit.name).with_suffix(".svg"))


if __name__ == "__main__":
    convert_circuit_directory_to_svg(Path("circuits"), Path("circuits_svg"))
