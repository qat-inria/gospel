from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
from graphix.opengraph import OpenGraph
from tqdm import tqdm

from gospel.brickwork_state_transpiler import (
    get_node_positions,
    layers_to_measurement_table,
    transpile,
    transpile_to_layers,
)
from gospel.scripts.qasm_parser import read_qasm

if TYPE_CHECKING:
    from graphix import Circuit, Pattern


def format_angle(angle: float) -> str:
    """
    Converts an angle in radians to a string representation as a multiple of π.
    """
    # If the angle is effectively zero, return "0"
    if abs(angle) < 1e-12:
        return "0"

    # Convert angle/π to a Fraction, limiting the denominator for a neat representation.
    frac = Fraction(angle).limit_denominator(1000)
    num, den = frac.numerator, frac.denominator

    # Determine the sign and work with absolute value for formatting.
    sign = "-" if num < 0 else ""
    num = abs(num)

    # When denominator is 1, we don't need to show it.
    if den == 1:
        if num == 1:
            return f"{sign}π"
        return f"{sign}{num}·π"
    if num == 1:
        return f"{sign}π/{den}"
    return f"{sign}{num}·π/{den}"


def draw_brickwork_state_pattern(pattern: Pattern, target: Path) -> None:
    graph = OpenGraph.from_pattern(pattern)
    pos = get_node_positions(pattern, reverse_qubit_order=True)
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

    pattern = transpile(circuit)
    graph = OpenGraph.from_pattern(pattern)
    pos = get_node_positions(pattern, reverse_qubit_order=True)
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


def draw_brickwork_state_colormap_from_pattern(
    pattern: Pattern,
    target: Path,
    failure_probas: dict[int, float],
    heavy_edges: set[tuple[int, int]] | None = None,
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

    graph = OpenGraph.from_pattern(pattern)
    pos = get_node_positions(pattern, reverse_qubit_order=True)
    labels = {node: node for node in graph.inside.nodes()}
    colors = [failure_probas[node] for node in graph.inside.nodes()]

    plt.figure(
        figsize=(max(x for x, y in pos.values()), max(y for x, y in pos.values()))
    )

    nx.draw_networkx_edges(graph.inside, pos, edge_color="black")

    # heavy edge overlay if provided
    if heavy_edges is not None:
        filtered_nodes: set[int] = set()

        for edge in heavy_edges:
            filtered_nodes.update(edge)  # should work
        filtered_pos = {i: j for i, j in pos.items() if i in filtered_nodes}

        heavy_graph = nx.Graph(heavy_edges)

        nx.draw_networkx_edges(heavy_graph, filtered_pos, edge_color="red", width=5)
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


def convert_circuit_directory_to_brickwork_state_svg(
    path_circuits: Path, path_brickwork_state_svg: Path
) -> None:
    path_brickwork_state_svg.mkdir()
    for path_circuit in tqdm(list(path_circuits.glob("*.qasm"))):
        with Path(path_circuit).open() as f:
            circuit = read_qasm(f)
            pattern = transpile(circuit)
            target = (path_brickwork_state_svg / path_circuit.name).with_suffix(".svg")
            draw_brickwork_state_pattern(pattern, target)


def convert_circuit_directory_to_brickwork_state_table(
    path_circuits: Path, path_brickwork_state_table: Path
) -> None:
    path_brickwork_state_table.mkdir()
    for path_circuit in tqdm(list(path_circuits.glob("*.qasm"))):
        with Path(path_circuit).open() as f:
            circuit = read_qasm(f)
            layers = transpile_to_layers(circuit)
            table_float = layers_to_measurement_table(layers)
            table_str = [
                [format_angle(angle / math.pi) for angle in column]
                for column in table_float
            ]
            target = (path_brickwork_state_table / path_circuit.name).with_suffix(
                ".json"
            )
            with target.open("w") as f_target:
                json.dump(table_str, f_target)


if __name__ == "__main__":
    convert_circuit_directory_to_brickwork_state_table(
        Path("pages/circuits"), Path("pages/brickwork_state_table")
    )
