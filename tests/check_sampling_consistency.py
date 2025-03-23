from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import graphix.command
from graphix.noise_models.noise_model import (
    CommandOrNoise,
    NoiseCommands,
    NoiseModel,
)
from graphix.rng import ensure_rng
from graphix.sim.statevec import StatevectorBackend
from veriphix.client import Client, Secrets

from gospel.scripts.qasm_parser import read_qasm

if TYPE_CHECKING:
    from graphix import Pattern
    from graphix.command import BaseM
    from numpy.random import Generator


BQP_ERROR = 0.4


def load_pattern_from_circuit(circuit_label: str) -> tuple[Pattern, list[int]]:
    with Path(f"circuits/{circuit_label}").open() as f:
        circuit = read_qasm(f)
        pattern = circuit.transpile().pattern
        # pattern = gospel.brickwork_state_transpiler.transpile(circuit)

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
        nodes: list[int],
        prob: float = 0.0,
        rng: Generator | None = None,
    ) -> None:
        self.prob = prob
        self.nodes = nodes
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
        return table[circuit_name], table[circuit_name] > 0.5


#######

outcomes_dict = {}
d = 100  # nr of computation rounds
num_instances = 100
# Load circuits list from the text file
with Path("gospel/cluster/sampled_circuits.txt").open() as f:
    instances = json.load(f)

# Noiseless simulation, only need to define the backend, no noise model
backend = StatevectorBackend()
for circuit in instances:
    # Generate a different instance
    pattern, onodes = load_pattern_from_circuit(circuit)

    # Instanciate Client and create Test runs
    client = Client(pattern=pattern, secrets=Secrets(a=True, r=True, theta=True))

    outcomes_sum_all_onodes = {onode: 0 for onode in onodes}
    for _i in range(d):
        # print("new round")
        client.delegate_pattern(backend=backend)
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
    print(f"{outcome}/100 -> simulation outcome: {majority_vote_outcome}")
    print(f"Expected outcome of majority vote: {int(expected_outcome)}")
