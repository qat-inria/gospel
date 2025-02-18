from dataclasses import dataclass

import networkx as nx
import stim
from graphix import Pattern, command
from graphix.clifford import Clifford
from graphix.command import CommandKind
from graphix.fundamentals import Axis, Sign
from graphix.measurements import PauliMeasurement
from graphix.pattern import pauli_nodes


def get_stabilizers(graph: nx.Graph) -> list[stim.PauliString]:
    """Method to generate the canonical stabilizers for a given graph state.

    :param graph: graph state
    :return: list of stim.Paulistring containing len(nodes) stabilizers
    """

    def get_stabilizer_for_node(node) -> stim.PauliString:
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
            raise ValueError(clifford)


def apply_pauli_measurement(
    sim: stim.TableauSimulator, node: int, measurement: PauliMeasurement
) -> bool:
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
    result = sim.measure(node)
    for clifford in cliffords:
        apply_clifford(sim, node, clifford)
    return result


@dataclass
class RenumberedGraph:
    nodes: list[int]
    edges: list[tuple[int, int]]
    renumbering: dict[int, int]
    graph: nx.Graph


def get_renumbered_graph(pattern: Pattern) -> RenumberedGraph:
    """Compute the graph state where nodes are indexed with a range of integers starting from 0.

    :param pattern: pattern
    :return: the renumbering and the graph
    """
    nodes, edges = pattern.get_graph()
    renumbering = {node: i for i, node in enumerate(nodes)}
    renumbered_edges = [(renumbering[u], renumbering[v]) for (u, v) in edges]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))
    graph.add_edges_from(renumbered_edges)
    return RenumberedGraph(nodes, edges, renumbering, graph)


def preprocess_pauli(pattern: Pattern, leave_input: bool) -> Pattern:
    pattern.move_pauli_measurements_to_the_front()
    renumbered_graph = get_renumbered_graph(pattern)
    stabilizers = get_stabilizers(renumbered_graph.graph)
    tableau = stim.Tableau.from_stabilizers(stabilizers)
    sim = stim.TableauSimulator()
    sim.do_tableau(tableau, list(renumbered_graph.graph.nodes))
    to_measure, non_pauli_meas = pauli_nodes(pattern, leave_input)
    results = {}
    for cmd, measurement in to_measure:
        node = renumbered_graph.renumbering[cmd.node]
        results[cmd.node] = apply_pauli_measurement(sim, node, measurement)
    tableau = sim.current_inverse_tableau().inverse()
    graph_state = tableau.to_circuit("graph_state")
    edges = []
    vops = {}
    for instruction in graph_state:
        match instruction.name:
            case "RX":
                pass
            case "CZ":
                for u, v in instruction.target_groups():
                    edges.append(
                        (
                            renumbered_graph.nodes[u.value],
                            renumbered_graph.nodes[v.value],
                        )
                    )
            case "H" | "S" | "X" | "Y" | "Z":
                clifford: Clifford = getattr(Clifford, instruction.name)
                for (u,) in instruction.target_groups():
                    node = renumbered_graph.nodes[u.value]
                    vops[node] = clifford @ vops.get(node, Clifford.I)
            case "TICK":
                pass
            case _:
                raise ValueError(instruction.name)
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
    for nodes in edges:
        result.add(command.E(nodes=nodes))
    for cmd in pattern:
        if cmd.kind == CommandKind.M and cmd.node in non_pauli_meas:
            vop = vops.get(cmd.node)
            if vop is not None:
                result.add(cmd.clifford(vop))
            else:
                result.add(cmd)
    for node in pattern.output_nodes:
        clifford = vops.get(node)
        if clifford is not None:
            result.add(command.C(node=node, clifford=clifford))
    return result
