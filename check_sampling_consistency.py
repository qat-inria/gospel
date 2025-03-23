from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import graphix.command
from graphix import Pattern
from graphix.noise_models.noise_model import (
    CommandOrNoise,
    NoiseCommands,
    NoiseModel,
)
from graphix.rng import ensure_rng
from graphix.sim.density_matrix import DensityMatrixBackend
from veriphix.client import Client, Secrets

import gospel.brickwork_state_transpiler
from gospel.scripts import read_qasm

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graphix.command import BaseM
    from numpy.random import Generator


BQP_ERROR = 0.4


def load_pattern_from_circuit(circuit_label: str) -> tuple[Pattern, list[int]]:
    with Path(f"circuits/{circuit_label}").open() as f:
        circuit = read_qasm(f)
        pattern = gospel.brickwork_state_transpiler.transpile(circuit)

        ## Measure output nodes, to have classical output
        classical_output = pattern.output_nodes
        for onode in classical_output:
            pattern.add(graphix.command.M(node=onode))

        # states = [BasicStates.PLUS] * len(pattern.input_nodes)

        # correct since the pattern is transpiled from a circuit and hence has a causal flow
        pattern.minimize_space()
    return pattern, classical_output


with Path("circuits/table.json").open() as f:
    table = json.load(f)
    circuits = [
        name for name, prob in table.items() if prob < BQP_ERROR or prob > 1 - BQP_ERROR
    ]


class GlobalNoiseModel(NoiseModel):
    """Global noise model.

    :param NoiseModel: Parent abstract class class:`graphix.noise_model.NoiseModel`
    :type NoiseModel: class
    """

    def __init__(
        self,
        nodes: Iterable[int],
        prob: float = 0.0,
        rng: Generator | None = None,
    ) -> None:
        self.prob = prob
        self.nodes = list(nodes)
        self.node = random.choice(self.nodes)
        self.rng = ensure_rng(rng)

    def refresh_randomness(self) -> None:
        self.node = random.choice(self.nodes)

    def input_nodes(self, nodes: list[int]) -> NoiseCommands:
        """Return the noise to apply to input nodes."""
        return []

    def command(self, cmd: CommandOrNoise) -> NoiseCommands:
        """Return the noise to apply to the command `cmd`."""
        return [cmd]

    def confuse_result(self, cmd: BaseM, result: bool) -> bool:
        """Assign wrong measurement result cmd = "M"."""
        if cmd.node == self.node and self.rng.uniform() < self.prob:
            return not result
        return result


def find_correct_value(circuit_name: str) -> tuple[float, bool]:
    with Path("circuits/table.json").open() as f:
        table = json.load(f)
        # return 1 if yes instance
        # return 0 else (no instance, as circuits are already filtered)
        # print(table[circuit_name])
        return float(table[circuit_name]), (table[circuit_name] > 0.5)


#######

p_err = 0
outcomes_dict = {}
d = 100  # nr of computation rounds
num_instances = 5
instances = random.sample(circuits, num_instances)

# Noiseless simulation, only need to define the backend, no noise model
backend = DensityMatrixBackend()
for circuit in instances:
    # Generate a different instance
    pattern, onodes = load_pattern_from_circuit(circuit)

    # Instanciate Client and create Test runs
    client = Client(pattern=pattern, secrets=Secrets(a=False, r=False, theta=False))

    outcomes_sum_all_onodes = dict.fromkeys(onodes, 0)
    noise_model = GlobalNoiseModel(prob=p_err, nodes=range(pattern.n_node))
    for _i in range(d):
        # print("new round")
        client.delegate_pattern(backend=backend, noise_model=noise_model)
        # Record outcomes of all output nodes
        for onode in onodes:
            outcomes_sum_all_onodes[onode] += client.results[onode]

    # Save the outcome of the first qubit by default
    outcomes_dict[circuit] = outcomes_sum_all_onodes[onodes[0]]
    outcome = int(outcomes_dict[circuit])
    majority_vote_outcome = "Ambig." if outcome == d / 2 else int(outcome > d / 2)
    p, expected_outcome = find_correct_value(circuit_name=circuit)
    print("#######")
    print(circuit)
    print(f"Prob. of getting 1: {p}")
    print(f"{outcome}/100 -> Noiseless outcome: {majority_vote_outcome}")
    print(f"Expected outcome of majority vote: {int(expected_outcome)}")
