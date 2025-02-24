import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from graphix import Circuit, Pattern, command
from graphix.opengraph import OpenGraph

import brickwork_state_transpiler


def add_j(pattern, src, tgt, angle):
    pattern.add(command.N(node=tgt))
    pattern.add(command.E(nodes=(src, tgt)))
    pattern.add(command.M(node=src, angle=angle))
    pattern.add(command.X(node=tgt, domain={src}))


def test_by_hand():
    circuit = Circuit(2)
    circuit.rz(0, 0.1)
    pattern = Pattern(input_nodes=[0, 1])
    add_j(pattern, 0, 2, -0.1 / math.pi)
    add_j(pattern, 1, 3, 0)
    add_j(pattern, 2, 4, 0)
    add_j(pattern, 3, 5, 0)
    pattern.add(command.E(nodes=(4, 5)))
    add_j(pattern, 4, 6, 0)
    add_j(pattern, 5, 7, 0)
    add_j(pattern, 6, 8, 0)
    add_j(pattern, 7, 9, 0)
    pattern.add(command.E(nodes=(8, 9)))

    graph = OpenGraph.from_pattern(pattern)
    pos = brickwork_state_transpiler.get_node_positions(pattern)
    labels = {
        node: measurement.angle for node, measurement in graph.measurements.items()
    }
    nx.draw(graph.inside, pos, labels=labels)
    plt.savefig("brickwork_state_graph.svg", format="svg")
    sv1 = circuit.simulate_statevector().statevec
    sv2 = pattern.simulate_pattern()
    assert np.abs(np.dot(sv1.flatten().conjugate(), sv2.flatten())) == pytest.approx(1)
