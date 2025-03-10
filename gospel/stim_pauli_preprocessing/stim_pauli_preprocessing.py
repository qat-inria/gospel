from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import stim
from graphix import Pattern, command
from graphix.clifford import Clifford
from graphix.command import CommandKind
from graphix.fundamentals import Axis, Plane, Sign
from graphix.measurements import Measurement, PauliMeasurement
from graphix.noise_models.depolarising_noise_model import (
    DepolarisingNoise,
    TwoQubitDepolarisingNoise,
)
from graphix.ops import Ops
from graphix.pattern import pauli_nodes
from graphix.sim.base_backend import Backend, BackendState
from graphix.simulator import DefaultMeasureMethod
from graphix.states import PlanarState, State

if TYPE_CHECKING:
    from collections.abc import Iterable

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


class StimBackend(Backend):  # type: ignore[misc]
    def __init__(
        self,
        sim: stim.TableauSimulator | None = None,
        branch: dict[int, bool] | None = None,
    ) -> None:
        super().__init__(BackendState())
        if sim is None:
            self.__sim = stim.TableauSimulator()
        else:
            self.__sim = sim
        self.__branch = branch

    @property
    def sim(self) -> stim.TableauSimulator:
        return self.__sim

    @property
    def branch(self) -> dict[int, bool] | None:
        return self.__branch

    def add_nodes(self, nodes: Iterable[int], data: State) -> None:
        if isinstance(data, PlanarState):
            if data.plane == Plane.XZ and data.angle == 0:
                return
            if data.plane == Plane.XY and data.angle == 0:
                self.sim.h(*nodes)
                return
        raise ValueError("Unsupported input state")

    def entangle_nodes(self, edge: tuple[int, int]) -> None:
        self.sim.cz(*edge)

    def measure(self, node: int, measurement: Measurement) -> bool:
        pm = PauliMeasurement.try_from(measurement.plane, measurement.angle / math.pi)
        if pm is None:
            raise ValueError(f"The measurement {measurement} is not in Pauli basis.")
        return apply_pauli_measurement(self.sim, node, pm, False, False, self.branch)

    def apply_single(self, node: int, op: Ops) -> None:
        if op is Ops.X:
            self.sim.x(node)
        elif op is Ops.Z:
            self.sim.z(node)
        else:
            raise ValueError(f"Unsupported operator: {op}")

    def apply_clifford(self, node: int, clifford: Clifford) -> None:
        apply_clifford(self.sim, node, clifford)

    def apply_noise(self, nodes: list[int], noise: Noise) -> None:
        match noise:
            case DepolarisingNoise(prob=prob):
                (q,) = nodes
                self.sim.depolarize1(q, p=prob)
            case TwoQubitDepolarisingNoise(prob=prob):
                (q0, q1) = nodes
                self.sim.depolarize2(q0, q1, p=prob)
            case _:
                raise ValueError(f"Unsupported noise: {noise}")

    def finalize(self, output_nodes: list[int]) -> None:
        pass


def simulate_pauli(
    sim: stim.TableauSimulator,
    pattern: Pattern,
    noise_model: NoiseModel | None = None,
    branch: dict[int, bool] | None = None,
) -> dict[int, bool]:
    backend = StimBackend(sim, branch)
    measure_method = DefaultMeasureMethod()
    pattern.simulate_pattern(
        backend, noise_model=noise_model, measure_method=measure_method
    )
    return measure_method.results  # type: ignore[no-any-return]


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
