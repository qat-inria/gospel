import math
import re
from io import TextIOBase
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from graphix import Circuit, Pattern
from graphix.opengraph import OpenGraph
from tqdm import tqdm

import gospel.brickwork_state_transpiler

command_re = re.compile(r"([a-z]+)(?:\(([^)]*)\))?")
reg_re = re.compile(r"[a-z0-9]+\[([^]]*)\]")


def parse_reg(s: str) -> int:
    reg = reg_re.fullmatch(s)
    if reg is None:
        raise ValueError(f"Invalid register: {s}")
    return int(reg.group(1))


def read_qasm(f: TextIOBase) -> Circuit:
    lines = "".join(line.strip() for line in f).split(";")
    if lines[0] != "OPENQASM 2.0":
        raise ValueError(f"Unexpected header: {lines[0]}")
    circuit = None
    for line in lines[1:]:
        if line == "":
            continue
        try:
            full_command, arguments_str = line.split(" ", 1)
        except ValueError:
            raise ValueError(f"Invalid syntax: {line}") from None
        arguments = arguments_str.split(",")
        command = command_re.fullmatch(full_command)
        if command is None:
            raise ValueError(f"Invalid syntax for command: {full_command}")
        command_name = command.group(1)
        match command_name:
            case "include":
                pass
            case "qreg":
                if len(arguments) != 1:
                    raise ValueError("qreg expects one argument")
                if circuit is not None:
                    raise ValueError("qreg cannot appear twice")
                circuit = Circuit(parse_reg(arguments[0]))
            case "creg":
                pass
            case "cx" | "rx" | "rz":
                if circuit is None:
                    raise ValueError("qreg is missing")
                if command_name == "cx":
                    if len(arguments) != 2:
                        raise ValueError("cx expects two arguments")
                    control = parse_reg(arguments[0])
                    target = parse_reg(arguments[1])
                    circuit.cnot(control, target)
                else:
                    angle = float(command.group(2))
                    if len(arguments) != 1:
                        raise ValueError(f"{command_name} expects one argument")
                    qubit = parse_reg(arguments[0])
                    if command_name == "rx":
                        circuit.rx(qubit, angle)
                    else:
                        circuit.rz(qubit, angle)
            case "measure":
                pass
            case _:
                raise ValueError(f"Unknown command: {command_name}")
    if circuit is None:
        raise ValueError("No circuit defined")
    return circuit


def format_angle(angle: float) -> str:
    if angle == 0:
        return "0"
    if angle == 1:
        return "π"
    if angle == -1:
        return "-π"
    if angle == 0.5:
        return "π/2"
    if angle == -0.5:
        return "-π/2"
    if angle == 0.25:
        return "π/4"
    if angle == -0.25:
        return "-π/4"
    return f"{angle * math.pi:.3f}"


def draw_brickwork_state_pattern(pattern: Pattern, target: Path) -> None:
    graph = OpenGraph.from_pattern(pattern)
    pos = gospel.brickwork_state_transpiler.get_node_positions(
        pattern, reverse_qubit_order=True
    )
    labels = {
        node: format_angle(measurement.angle)
        for node, measurement in graph.measurements.items()
    }
    plt.figure(
        figsize=(max(x for x, y in pos.values()), max(y for x, y in pos.values()))
    )
    nx.draw(
        graph.inside,
        pos,
        labels=labels,
        node_size=1000,
        node_color="white",
        edgecolors="black",
        font_size=9,
    )
    plt.savefig(target, format="svg")
    plt.close()


def draw_brickwork_state_colormap(
    circuit: Circuit, target: Path, failure_probas: dict[int, float]
) -> None:
    """Draw the brickwork state with trap failure probability drawn as the node color.
    Heavily redundant since we have already gone through the transpilation step.

    Parameters
    ----------
    circuit : Circuit

    target : Path
        where to save the figure
    failure_probas : dict[int, float]
        dictionary of failure probability (value) by node (key)
    """

    pattern = gospel.brickwork_state_transpiler.transpile(circuit)
    graph = OpenGraph.from_pattern(pattern)
    pos = gospel.brickwork_state_transpiler.get_node_positions(
        pattern, reverse_qubit_order=True
    )
    labels = {node: node for node in graph.inside.nodes()}
    colors = [failure_probas[node] for node in graph.inside.nodes()]

    plt.figure(
        figsize=(max(x for x, y in pos.values()), max(y for x, y in pos.values()))
    )
    nx.draw_networkx_edges(graph.inside, pos, edge_color="black")
    # false error: Argument "node_color" to "draw_networkx_nodes" has incompatible type "list[float]"; expected "str"  [arg-type]
    # false error: Module has no attribute "jet"  [attr-defined]
    nc = nx.draw_networkx_nodes(
        graph.inside,
        pos,
        nodelist=graph.inside.nodes,
        label=labels,
        node_color=colors,  # type: ignore[arg-type]
        node_size=1000,
        cmap=plt.cm.jet,  # type: ignore[attr-defined]
        vmin=0,
        vmax=1,
    )
    plt.colorbar(nc)
    plt.axis("off")
    plt.savefig(target, format="svg")
    plt.close()


def convert_circuit_directory_to_brickwork_state(
    path_circuits: Path, path_brickwork_state_svg: Path
) -> None:
    path_brickwork_state_svg.mkdir()
    for path_circuit in tqdm(list(path_circuits.glob("*.qasm"))):
        with Path(path_circuit).open() as f:
            circuit = read_qasm(f)
            pattern = gospel.brickwork_state_transpiler.transpile(circuit)
            target = (path_brickwork_state_svg / path_circuit.name).with_suffix(".svg")
            draw_brickwork_state_pattern(pattern, target)


if __name__ == "__main__":
    convert_circuit_directory_to_brickwork_state(
        Path("pages/circuits"), Path("pages/brickwork_state_svg")
    )
