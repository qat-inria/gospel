from __future__ import annotations

import enum
import math
import typing
from abc import ABC, abstractmethod
from array import array
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from graphix import Pattern, command
from graphix.instruction import InstructionKind

if TYPE_CHECKING:
    from graphix import Circuit, instruction


class Brick(ABC):
    @abstractmethod
    def measures(self) -> list[list[float]]: ...


class CNOT(Brick):
    def measures(self) -> list[list[float]]:
        return [[0, 0, math.pi / 4, 0], [0, math.pi / 4, 0, -math.pi / 4]]


class XZ(Enum):
    X = enum.auto()
    Z = enum.auto()


def value_or_zero(v: float | None) -> float:
    if v is None:
        return 0
    return v


@dataclass
class SingleQubit:
    rz0: float | None = None
    rx: float | None = None
    rz1: float | None = None

    def measures(self) -> list[float]:
        return [
            -value_or_zero(self.rz0),
            -value_or_zero(self.rx),
            -value_or_zero(self.rz1),
            0,
        ]

    def is_identity(self) -> bool:
        return self.rz0 is None and self.rx is None and self.rz1 is None

    def add(self, axis: XZ, angle: float) -> bool:
        match axis:
            case XZ.X:
                if self.rx is None and self.rz1 is None:
                    self.rx = angle
                    return True
                return False
            case XZ.Z:
                if self.rz0 is None and self.rx is None:
                    self.rz0 = angle
                    return True
                if self.rz1 is None:
                    self.rz1 = angle
                    return True
                return False
            case _:
                typing.assert_never(axis)


@dataclass
class SingleQubitPair(Brick):
    top: SingleQubit
    bottom: SingleQubit

    def get(self, position: bool) -> SingleQubit:
        if position:
            return self.bottom
        return self.top

    def measures(self) -> list[list[float]]:
        return [self.top.measures(), self.bottom.measures()]


@dataclass
class Layer:
    odd: bool
    bricks: list[Brick]

    def get(self, qubit: int) -> tuple[Brick, bool]:
        index = (qubit - int(self.odd)) // 2
        return (self.bricks[index], bool(qubit % 2) != self.odd)


CNot = CNOT()


def __get_layer(width: int, layers: list[Layer], depth: int) -> Layer:
    for i in range(len(layers), depth + 1):
        odd = bool(i % 2)
        layer_size = (width - 1) // 2 if odd else (width + 1) // 2
        layers.append(
            Layer(
                odd,
                [
                    SingleQubitPair(SingleQubit(), SingleQubit())
                    for _ in range(layer_size)
                ],
            )
        )
    return layers[depth]


def __insert_rotation(
    width: int,
    layers: list[Layer],
    depth: list[int],
    instr: instruction.RX | instruction.RZ,
) -> None:
    axis = XZ.X if instr.kind == InstructionKind.RX else XZ.Z
    target_depth = depth[instr.target]
    if target_depth > 0:
        previous_layer = layers[target_depth - 1]
        brick, position = previous_layer.get(instr.target)
        if isinstance(brick, SingleQubitPair):
            gate = brick.get(position)
            if gate.add(axis, instr.angle):
                return
        else:
            assert brick is CNot
    if target_depth % 2 and (
        instr.target == 0 or (width % 2 == 0 and instr.target == width - 1)
    ):
        target_depth += 1
    layer = __get_layer(width, layers, target_depth)
    brick, position = layer.get(instr.target)
    assert isinstance(brick, SingleQubitPair)
    gate = brick.get(position)
    assert gate.is_identity()
    added = gate.add(axis, instr.angle)
    assert added
    depth[instr.target] = target_depth + 1


def transpile_to_layers(circuit: Circuit) -> list[Layer]:
    layers = []
    depth = [0 for _ in range(circuit.width)]
    for instr in circuit.instruction:
        match instr.kind:
            case InstructionKind.CNOT:
                if abs(instr.control - instr.target) != 1:
                    raise ValueError(
                        "Unsupported CNOT: control and target qubits should be consecutive"
                    )
                target = min(instr.control, instr.target)
                min_depth = max(depth[target], depth[target + 1])
                target_depth = (
                    min_depth if target % 2 == min_depth % 2 else min_depth + 1
                )
                target_layer = __get_layer(circuit.width, layers, target_depth)
                index = target // 2
                target_layer.bricks[index] = CNot
                depth[target] = target_depth + 1
                depth[target + 1] = target_depth + 1
            case InstructionKind.RX | InstructionKind.RZ:
                __insert_rotation(circuit.width, layers, depth, instr)
            case _:
                raise ValueError(
                    "Unsupported gate: circuits should contain only CNOT, RX and RZ"
                )
    return layers


@dataclass
class NodeGenerator:
    pattern: Pattern
    from_index: int

    def fresh(self) -> int:
        index = self.from_index
        self.from_index += 1
        self.pattern.add(command.N(node=index))
        return index


def add_j(
    pattern: Pattern, node_generator: NodeGenerator, node: int, angle: float
) -> int:
    next_node = node_generator.fresh()
    pattern.add(command.E(nodes=(node, next_node)))
    pattern.add(command.M(node=node, angle=angle / math.pi))
    pattern.add(command.X(node=next_node, domain={node}))
    return next_node


def layers_to_pattern(width: int, layers: list[Layer]) -> Pattern:
    input_nodes = list(range(width))
    pattern = Pattern(input_nodes)
    nodes = input_nodes
    node_generator = NodeGenerator(pattern, width)
    if width % 2:
        nodes.append(node_generator.fresh())
        last_qubit = width
    else:
        last_qubit = width - 1
    for layer in layers:
        all_brick_measures = [brick.measures() for brick in layer.bricks]
        for col in range(4):
            if layer.odd:
                print(list(pattern), nodes[0])
                nodes[0] = add_j(pattern, node_generator, nodes[0], 0)
            qubit = int(layer.odd)
            for measures in all_brick_measures:
                nodes[qubit] = add_j(
                    pattern, node_generator, nodes[qubit], measures[0][col]
                )
                nodes[qubit + 1] = add_j(
                    pattern, node_generator, nodes[qubit + 1], measures[1][col]
                )
                if col in {1, 3}:
                    pattern.add(command.E(nodes=(nodes[qubit], nodes[qubit + 1])))
                qubit += 2
            if layer.odd:
                nodes[last_qubit] = add_j(pattern, node_generator, nodes[last_qubit], 0)
    if width % 2:
        pattern.add(command.M(node=last_qubit, angle=0))
    return pattern


def transpile(circuit: Circuit) -> Pattern:
    layers = transpile_to_layers(circuit)
    return layers_to_pattern(circuit.width, layers)


def get_node_positions(pattern: Pattern, scale: float = 1) -> dict[int, array]:
    width = len(pattern.input_nodes)
    if width % 2:
        width = width + 1
    assert pattern.n_node % width == 0
    return {
        node: array("i", [(node // width) * scale, (width - node % width) * scale])
        for node in range(pattern.n_node)
    }
