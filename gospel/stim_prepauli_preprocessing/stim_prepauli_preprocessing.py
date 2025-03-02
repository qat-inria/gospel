from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

import networkx as nx
import stim
from graphix import Pattern, command
from graphix.clifford import Clifford
from graphix.command import CommandKind
from graphix.fundamentals import Axis, Sign
from graphix.measurements import PauliMeasurement
from graphix.noise_models.depolarising_noise_model import (
    DepolarisingNoiseElement,
    TwoQubitDepolarisingNoiseElement,
)
from graphix.noise_models.noiseless_noise_model import NoiselessNoiseModel
from graphix.pattern import pauli_nodes

if TYPE_CHECKING:
    from graphix.noise_models.noise_model import Noise, NoiseModel

if TYPE_CHECKING:
    GraphType = nx.Graph[int]
else:
    GraphType = nx.Graph


def get_stabilizers(graph: GraphType) -> list[stim.PauliString]:
    """Method to generate the canonical stabilizers for a given graph state.

    :param graph: graph state
    :return: list of stim.Paulistring containing len(nodes) stabilizers
    """

    def get_stabilizer_for_node(node: int) -> stim.PauliString:
        ps = stim.PauliString(graph.number_of_nodes())
        ps[node] = "X"
        for k in graph.neighbors(node):
            ps[k] = "Z"
        return ps

    return [get_stabilizer_for_node(node) for node in graph.nodes]


def apply_clifford(sim: stim.TableauSimulator, node: int, clifford: Clifford) -> None:
    match clifford:
        case Clifford.H:
            sim.h(node)
        case Clifford.S:
            sim.s(node)
        case Clifford.SDG:
            sim.s_dag(node)
        case Clifford.Z:
            sim.z(node)
        case clifford.X:
            sim.x(node)
        case _:
            for h_s_z in clifford.hsz:
                match h_s_z:
                    case Clifford.H:
                        sim.h(node)
                    case Clifford.S:
                        sim.s(node)
                    case Clifford.Z:
                        sim.z(node)
                    case _:
                        raise ValueError("Unreachable")


def apply_pauli_measurement(
    sim: stim.TableauSimulator,
    node: int,
    measurement: PauliMeasurement,
    s_signal: bool,
    t_signal: bool,
    branch: dict[int, bool] | None = None,
) -> bool:
    if s_signal:
        sim.h(node)
        sim.z(node)
        sim.h(node)
    if t_signal:
        sim.z(node)
    match measurement.sign, measurement.axis:
        case Sign.PLUS, Axis.X:
            cliffords = [Clifford.H]
        case Sign.MINUS, Axis.X:
            cliffords = [Clifford.H, Clifford.Z]
        case Sign.PLUS, Axis.Y:
            cliffords = [Clifford.H, Clifford.S]
        case Sign.MINUS, Axis.Y:
            cliffords = [Clifford.H, Clifford.S, Clifford.Z]
        case Sign.PLUS, Axis.Z:
            cliffords = []
        case Sign.MINUS, Axis.Z:
            cliffords = [Clifford.X]
        case _:
            raise ValueError("unreachable")
    for clifford in reversed(cliffords):
        apply_clifford(sim, node, clifford.conj)
    branch_result = None if branch is None else branch.get(node)
    if branch_result is None:
        result = sim.measure(node)
    else:
        result = branch_result
        sim.postselect_z(node, desired_value=result)
    for clifford in cliffords:
        apply_clifford(sim, node, clifford)
    return result


@dataclass
class RenumberedGraph:
    nodes: list[int]
    edges: list[tuple[int, int]]
    renumbering: dict[int, int]
    graph: GraphType


def get_renumbered_graph(pattern: Pattern) -> RenumberedGraph:
    """Compute the graph state where nodes are indexed with a range of integers starting from 0.

    :param pattern: pattern
    :return: the renumbering and the graph
    """
    nodes, edges = pattern.get_graph()
    renumbering = {node: i for i, node in enumerate(nodes)}
    renumbered_edges = [(renumbering[u], renumbering[v]) for (u, v) in edges]
    graph: GraphType = GraphType()
    graph.add_nodes_from(range(len(nodes)))
    graph.add_edges_from(renumbered_edges)
    return RenumberedGraph(nodes, edges, renumbering, graph)


def graph_state_to_edges_and_vops(
    graph_state: stim.Circuit,
) -> tuple[list[tuple[int, int]], dict[int, Clifford]]:
    edges: list[tuple[int, int]] = []
    vops: dict[int, Clifford] = {}
    # "Circuit" has no attribute "__iter__"
    # (but __len__ and __getitem__)
    for instruction in graph_state:  # type: ignore[attr-defined]
        match instruction.name:
            case "RX":
                pass
            case "CZ":
                edges.extend((u.value, v.value) for u, v in instruction.target_groups())
            case "H" | "S" | "X" | "Y" | "Z":
                clifford: Clifford = getattr(Clifford, instruction.name)
                for (u,) in instruction.target_groups():
                    vops[u.value] = clifford @ vops.get(u.value, Clifford.I)
            case "TICK":
                pass
            case _:
                raise ValueError(instruction.name)
    return edges, vops


def apply_pauli_noise(sim: stim.TableauSimulator, noise: Noise) -> None:
    for element, qubits in noise:
        match element:
            case DepolarisingNoiseElement(prob=prob):
                (q,) = qubits
                sim.depolarize1(q, p=prob)
            case TwoQubitDepolarisingNoiseElement(prob=prob):
                (q0, q1) = qubits
                sim.depolarize2(q0, q1, p=prob)
            case _:
                raise ValueError(f"Unsupported noise element: {element}")


def cut_pattern(pattern: Pattern) -> tuple[Pattern, Pattern]:
    pauli_pattern = Pattern(input_nodes=pattern.input_nodes)
    it = iter(pattern)
    for cmd in it:
        if (
            cmd.kind == CommandKind.M
            and PauliMeasurement.try_from(cmd.plane, cmd.angle) is None
        ):
            break
        pauli_pattern.add(cmd)
    non_pauli_pattern = Pattern(input_nodes=pauli_pattern.output_nodes)
    non_pauli_pattern.add(cmd)
    non_pauli_pattern.extend(it)
    return (pauli_pattern, non_pauli_pattern)


def simulate_pauli(
    sim: stim.TableauSimulator,
    pattern: Pattern,
    noise_model: NoiseModel | None = None,
    branch: dict[int, bool] | None = None,
) -> dict[int, bool]:
    if noise_model is None:
        noise_model = NoiselessNoiseModel()
    results: dict[int, bool] = {}
    for node in pattern.input_nodes:
        sim.h(node)
    for cmd in pattern:
        # Use of `if` instead of `match` here for mypy
        if cmd.kind == CommandKind.N:
            sim.h(cmd.node)
        elif cmd.kind == CommandKind.E:
            sim.cz(*cmd.nodes)
        elif cmd.kind == CommandKind.M:
            pm = PauliMeasurement.try_from(cmd.plane, cmd.angle)
            if pm is None:
                raise ValueError(f"The measurement {cmd} is not in Pauli basis.")
            apply_pauli_noise(sim, noise_model.command(cmd))
            s_signal = bool(sum(results[node] for node in cmd.s_domain) % 2)
            t_signal = bool(sum(results[node] for node in cmd.t_domain) % 2)
            result = apply_pauli_measurement(
                sim, cmd.node, pm, s_signal, t_signal, branch
            )
            result = noise_model.confuse_result(result)
            results[cmd.node] = result
        # Use of `==` here for mypy
        elif cmd.kind == CommandKind.X or cmd.kind == CommandKind.Z:  # noqa: PLR1714
            if sum(results[node] for node in cmd.domain) % 2:
                if cmd.kind == CommandKind.X:
                    sim.x(cmd.node)
                else:
                    sim.z(cmd.node)
        elif cmd.kind == CommandKind.C:
            apply_clifford(sim, cmd.node, cmd.clifford)
        elif cmd.kind == CommandKind.T:
            pass
        elif cmd.kind == CommandKind.S:
            raise ValueError("Unexpected S gate")
        else:
            assert_never(cmd.kind)
        if cmd.kind != CommandKind.M:
            apply_pauli_noise(sim, noise_model.command(cmd))
    return results


def graph_state_to_pattern(
    circuit: stim.Circuit, input_nodes: list[int], output_nodes: list[int]
) -> Pattern:
    edges, vops = graph_state_to_edges_and_vops(circuit)
    pattern = Pattern(input_nodes)
    input_node_set = set(input_nodes)
    output_node_set = set(output_nodes)
    pattern.extend(
        command.N(node=node) for node in output_nodes if node not in input_node_set
    )
    pattern.extend(command.E(nodes=nodes) for nodes in edges)
    for node, clifford in vops.items():
        print(node, clifford)
    pattern.extend(
        command.C(node=node, clifford=clifford)
        for node, clifford in vops.items()
        if node in output_node_set
    )
    return pattern


def preprocess_pauli(
    pattern: Pattern, leave_input: bool, branch: dict[int, bool] | None = None
) -> Pattern:
    pattern.move_pauli_measurements_to_the_front()
    renumbered_graph = get_renumbered_graph(pattern)
    stabilizers = get_stabilizers(renumbered_graph.graph)
    tableau = stim.Tableau.from_stabilizers(stabilizers)
    sim = stim.TableauSimulator()
    sim.do_tableau(tableau, list(renumbered_graph.graph.nodes))
    to_measure, non_pauli_meas = pauli_nodes(pattern, leave_input)
    results: dict[int, bool] = {}
    for m, measurement in to_measure:
        node = renumbered_graph.renumbering[m.node]
        s_signal = bool(sum(results[node] for node in m.s_domain) % 2)
        t_signal = bool(sum(results[node] for node in m.t_domain) % 2)
        results[m.node] = apply_pauli_measurement(
            sim, node, measurement, s_signal, t_signal, branch
        )
    tableau = sim.current_inverse_tableau().inverse()
    graph_state = tableau.to_circuit("graph_state")
    edges, vops = graph_state_to_edges_and_vops(graph_state)
    if leave_input:
        input_nodes = pattern.input_nodes
    else:
        input_nodes = [node for node in pattern.input_nodes if node in non_pauli_meas]
    input_node_set = set(input_nodes)
    result = Pattern(input_nodes)
    result.results = results
    for node in renumbered_graph.nodes:
        if node not in results and node not in input_node_set:
            result.add(command.N(node=node))
    for u, v in edges:
        result.add(
            command.E(nodes=(renumbered_graph.nodes[u], renumbered_graph.nodes[v]))
        )
    for cmd in pattern:
        if cmd.kind == CommandKind.M and cmd.node in non_pauli_meas:
            vop = vops.get(renumbered_graph.renumbering[cmd.node])
            if vop is not None:
                print(cmd.node, vop)
                result.add(cmd.clifford(vop))
            else:
                result.add(cmd)
        elif cmd.kind in (CommandKind.Z, CommandKind.X, CommandKind.C):
            result.add(cmd)
    for node in pattern.output_nodes:
        clifford: Clifford | None = vops.get(renumbered_graph.renumbering[node])
        if clifford is not None:
            result.add(command.C(node=node, clifford=clifford))
    result.reorder_output_nodes(pattern.output_nodes)
    return result
