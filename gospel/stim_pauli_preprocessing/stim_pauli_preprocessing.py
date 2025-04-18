from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

import networkx as nx
import stim
from graphix import Pattern, command
from graphix.clifford import Clifford
from graphix.command import CommandKind
from graphix.fundamentals import Axis, Sign
from graphix.measurements import Measurement, PauliMeasurement
from graphix.noise_models.depolarising_noise_model import (
    DepolarisingNoise,
    TwoQubitDepolarisingNoise,
)
from graphix.ops import Ops
from graphix.pattern import pauli_nodes
from graphix.sim.base_backend import Backend, BackendState
from graphix.sim.statevec import Statevec
from graphix.simulator import DefaultMeasureMethod
from graphix.states import BasicState, BasicStates, State

from gospel.noise_models.single_pauli_noise_model import SinglePauliNoise

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    import numpy.typing as npt
    from graphix.noise_models.noise_model import Noise, NoiseCommands, NoiseModel

if TYPE_CHECKING:
    GraphType = nx.Graph[int]
else:
    GraphType = nx.Graph


BASIC_STATE_TO_CLIFFORD = {
    BasicState.ZERO: [Clifford.Z],
    BasicState.ONE: [Clifford.X],
    BasicState.PLUS: [Clifford.H],
    BasicState.MINUS: [Clifford.H, Clifford.Z],
    BasicState.PLUS_I: [Clifford.H, Clifford.S],
    BasicState.MINUS_I: [Clifford.H, Clifford.S, Clifford.Z],
}


def basic_state_to_clifford_gates(basic_state: BasicState) -> list[Clifford]:
    return BASIC_STATE_TO_CLIFFORD[basic_state]


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


def pauli_measurement_to_clifford_gates(
    measurement: PauliMeasurement,
) -> list[Clifford]:
    match measurement.sign, measurement.axis:
        case Sign.PLUS, Axis.X:
            return [Clifford.H]
        case Sign.MINUS, Axis.X:
            return [Clifford.H, Clifford.Z]
        case Sign.PLUS, Axis.Y:
            return [Clifford.H, Clifford.S]
        case Sign.MINUS, Axis.Y:
            return [Clifford.H, Clifford.S, Clifford.Z]
        case Sign.PLUS, Axis.Z:
            return []
        case Sign.MINUS, Axis.Z:
            return [Clifford.X]
        case _:
            raise ValueError("unreachable")


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
    cliffords = pauli_measurement_to_clifford_gates(measurement)
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


class StimBackendState(BackendState):
    def flatten(self) -> npt.NDArray[np.complex128]:
        """Return flattened state."""
        return NotImplemented


class StimBackend(Backend):
    def __init__(
        self,
        sim: stim.TableauSimulator | None = None,
        branch: dict[int, bool] | None = None,
    ) -> None:
        super().__init__(StimBackendState())
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

    def add_nodes(
        self, nodes: Iterable[int], data: Statevec | State = BasicStates.PLUS
    ) -> None:
        state = BasicState.try_from_statevector(Statevec(data).psi)

        if state is None:
            raise ValueError(
                f"Incorrect state value: stim can only prepare stabiliser states {data}."
            )

        if state == BasicState.ZERO:
            # required by stim otherwise empty stabiliser
            self.sim.z(*nodes)
        elif state == BasicState.ONE:
            self.sim.x(*nodes)
        elif state == BasicState.PLUS:
            self.sim.h(*nodes)
        elif state == BasicState.MINUS:
            self.sim.h(*nodes)
            self.sim.z(*nodes)
        elif state == BasicState.PLUS_I:
            self.sim.h(*nodes)
            self.sim.s(*nodes)
        elif state == BasicState.MINUS_I:
            self.sim.h(*nodes)
            self.sim.s(*nodes)
            self.sim.z(*nodes)
        else:
            assert_never(state)

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
            # add case here
            case _:
                raise ValueError(f"Unsupported noise: {noise} and {nodes}")

    def finalize(self, output_nodes: list[int]) -> None:
        pass

    def to_pattern(self, input_nodes: list[int], output_nodes: list[int]) -> Pattern:
        tableau = self.sim.current_inverse_tableau().inverse()
        circuit = tableau.to_circuit("graph_state")
        return graph_state_to_pattern(circuit, input_nodes, output_nodes)


def pattern_to_stim_circuit(
    pattern: Pattern,
    noise_model: NoiseModel | None = None,
    input_state: dict[int, BasicState] | BasicState = BasicState.PLUS,
    fixed_states: dict[int, BasicState] | None = None,
) -> tuple[stim.Circuit, dict[int, int]]:
    circuit = stim.Circuit()
    if isinstance(input_state, BasicState):
        for clifford in basic_state_to_clifford_gates(input_state):
            circuit.append(str(clifford), targets=pattern.input_nodes)  # type: ignore[call-overload]
    else:
        other_nodes = set(input_state.keys()) - set(pattern.input_nodes)
        if other_nodes:
            raise ValueError(f"Not input states: {other_nodes}")
        for node in pattern.input_nodes:
            basic_state = input_state[node]
            for clifford in basic_state_to_clifford_gates(basic_state):
                circuit.append(str(clifford), targets=[node])  # type: ignore[call-overload]
    if noise_model is None:
        actual_pattern: NoiseCommands = list(pattern)
    else:
        actual_pattern = noise_model.input_nodes(pattern.input_nodes)
        actual_pattern.extend(noise_model.transpile(list(pattern)))
    measure_count = 0
    measure_indices: dict[int, int] = {}

    def get_target(node: int) -> stim.GateTarget:
        return stim.target_rec(measure_indices[node] - measure_count)

    for cmd in actual_pattern:
        if cmd.kind == CommandKind.N:
            if fixed_states is not None:
                basic_state_or_none = fixed_states.get(cmd.node)
            else:
                basic_state_or_none = None
            if basic_state_or_none is None:
                basic_state_or_none = BasicState.try_from_statevector(
                    Statevec(cmd.state).psi
                )
                if basic_state_or_none is None:
                    raise ValueError(f"Non-Pauli preparation: {cmd}")
            for clifford in basic_state_to_clifford_gates(basic_state_or_none):
                circuit.append(str(clifford), [cmd.node])  # type: ignore[call-overload]
        elif cmd.kind == CommandKind.E:
            circuit.append("CZ", cmd.nodes)  # type: ignore[call-overload]
        elif cmd.kind == CommandKind.M:
            for node in cmd.s_domain:
                circuit.append("CX", [get_target(node), cmd.node])  # type: ignore[call-overload]
            for node in cmd.t_domain:
                circuit.append("CZ", [get_target(node), cmd.node])  # type: ignore[call-overload]
            measurement = PauliMeasurement.try_from(cmd.plane, cmd.angle)
            if measurement is None:
                raise ValueError(f"Non-Pauli measurement: {cmd}")
            cliffords = pauli_measurement_to_clifford_gates(measurement)
            for clifford in reversed(cliffords):
                circuit.append(str(clifford), [cmd.node])  # type: ignore[call-overload]
            circuit.append("M", [cmd.node])  # type: ignore[call-overload]
            for clifford in cliffords:
                circuit.append(str(clifford), [cmd.node])  # type: ignore[call-overload]
            measure_indices[cmd.node] = measure_count
            measure_count += 1
        elif cmd.kind == CommandKind.X:
            for node in cmd.domain:
                circuit.append("CX", [get_target(node), cmd.node])  # type: ignore[call-overload]
        elif cmd.kind == CommandKind.Z:
            for node in cmd.domain:
                circuit.append("CZ", [get_target(node), cmd.node])  # type: ignore[call-overload]
        elif cmd.kind == CommandKind.C:
            circuit.append(str(cmd.clifford), [cmd.node])  # type: ignore[call-overload]
        elif cmd.kind == CommandKind.A:
            match cmd.noise:
                case DepolarisingNoise(prob=prob):
                    (q,) = cmd.nodes
                    circuit.append("DEPOLARIZE1", [q], prob)
                case TwoQubitDepolarisingNoise(prob=prob):
                    (q0, q1) = cmd.nodes
                    circuit.append("DEPOLARIZE2", [q0, q1], prob)
                # add case here
                case SinglePauliNoise(
                    prob=prob, error_type=et
                ):  #  case SinglePauliNoise(error_type='X')
                    (q,) = cmd.nodes
                    if et == "X":
                        circuit.append("X_ERROR", [q], prob)
                    elif et == "Z":
                        circuit.append("Z_ERROR", [q], prob)
                    else:
                        raise ValueError(
                            f"Unsupported single-Pauli: {et} and {cmd.nodes}"
                        )
                    # deterministic case just a gate
                    # circuit.append(et, [q])
                case _:
                    raise ValueError(f"Unsupported noise: {cmd.noise} and {cmd.nodes}")

    return circuit, measure_indices


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
    return measure_method.results


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
