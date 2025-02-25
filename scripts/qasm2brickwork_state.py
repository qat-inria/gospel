import math
import re
from io import TextIOBase
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from graphix import Circuit
from graphix.opengraph import OpenGraph

import brickwork_state_transpiler

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
    return circuit


def format_angle(angle) -> str:
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


def draw_brickwork_state(circuit: Circuit, target: Path) -> None:
    pattern = brickwork_state_transpiler.transpile(circuit)
    graph = OpenGraph.from_pattern(pattern)
    pos = brickwork_state_transpiler.get_node_positions(pattern)
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


def convert_circuit_directory():
    circuits_path = Path("pages/circuits")
    circuits_svg_path = Path("pages/brickwork_state_svg")
    circuits_svg_path.mkdir()
    for circuit_path in list(circuits_path.glob("*.qasm")):
        with Path(circuit_path).open() as f:
            circuit = read_qasm(f)
            target = (circuits_svg_path / circuit_path.name).with_suffix(".svg")
            draw_brickwork_state(circuit, target)


if __name__ == "__main__":
    convert_circuit_directory()
