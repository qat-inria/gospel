from __future__ import annotations

import random
from pathlib import Path

import graphix.command
import numpy as np
import stim
import veriphix.client
from graphix.fundamentals import IXYZ
from graphix.noise_models import DepolarisingNoiseModel, NoiseModel
from graphix.pauli import Pauli
from graphix.random_objects import Circuit, rand_circuit
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import Statevec, StatevectorBackend
from graphix.states import BasicStates
from veriphix.client import CircuitUtils, Client, Secrets, TrappifiedCanvas
import json
from pathlib import Path
import gospel.brickwork_state_transpiler
from gospel.scripts.qasm2brickwork_state import read_qasm, draw_brickwork_state_colormap


"""Global noise model."""

from typing import TYPE_CHECKING

import typing_extensions

from graphix.channels import KrausChannel, dephasing_channel
from graphix.command import Command, CommandKind, BaseM
from graphix.noise_models.noise_model import Noise, NoiseModel, NoiseCommands, CommandOrNoise
from graphix.rng import ensure_rng
import random


if TYPE_CHECKING:
    from numpy.random import Generator




BQP_ERROR=0.4


def load_pattern_from_circuit(circuit_label:str):
    with Path(f"circuits/{circuit_label}").open() as f:
        circuit = read_qasm(f)
        pattern = gospel.brickwork_state_transpiler.transpile(circuit)

        ## Measure output nodes, to have classical output
        classical_output = pattern.output_nodes
        for onode in classical_output:
            pattern.add(graphix.command.M(node=onode))

        states = [BasicStates.PLUS] * len(pattern.input_nodes)

        # correct since the pattern is transpiled from a circuit and hence has a causal flow
        pattern.minimize_space()
    return pattern, classical_output

with Path("circuits/table.json").open() as f:
    table = json.load(f)
    circuits = [name for name,prob in table.items() if prob < BQP_ERROR or prob > 1-BQP_ERROR]



class GlobalNoiseModel(NoiseModel):
    """Global noise model.

    :param NoiseModel: Parent abstract class class:`graphix.noise_model.NoiseModel`
    :type NoiseModel: class
    """

    def __init__(
        self,
        nodes: list[int],
        prob: float = 0.0,
        rng: Generator = None,
    ) -> None:
        self.prob = prob
        self.nodes = nodes
        self.node = random.choice(self.nodes)
        self.rng = ensure_rng(rng)

    def refresh_randomness(self):
        self.node = random.choice(self.nodes)

    def input_nodes(self, nodes: list[int]) -> NoiseCommands:
        """Return the noise to apply to input nodes."""
        return []

    def command(self, cmd: CommandOrNoise) -> NoiseCommands:
        """Return the noise to apply to the command `cmd`."""
        return [cmd]

    def confuse_result(self, cmd:BaseM, result: bool) -> bool:
        """Assign wrong measurement result cmd = "M"."""
        if cmd.node == self.node and self.rng.uniform() < self.prob:
            return not result
        else:
            return result
    

def find_correct_value(circuit_name):
    with Path("circuits/table.json").open() as f:
        table = json.load(f)
        # return 1 if yes instance
        # return 0 else (no instance, as circuits are already filtered)
        # print(table[circuit_name])
        return table[circuit_name], (int(table[circuit_name] > 0.5))
#######
        
p_err = 0
outcomes_dict = {}
d = 100       # nr of computation rounds
num_instances = 5
instances = random.sample(circuits, num_instances)

# Noiseless simulation, only need to define the backend, no noise model
backend = DensityMatrixBackend()
for circuit in instances:
    # Generate a different instance
    pattern, onodes = load_pattern_from_circuit(circuit)

    # Instanciate Client and create Test runs
    client = Client(pattern=pattern, secrets=Secrets(a=False, r=False, theta=False))

    outcomes_sum_all_onodes = {onode:0 for onode in onodes}
    noise_model = GlobalNoiseModel(prob=p_err, nodes=range(pattern.n_node))
    for i in range(d):
        # print("new round")
        client.delegate_pattern(backend=backend, noise_model=noise_model)
        # Record outcomes of all output nodes
        for onode in onodes:
            outcomes_sum_all_onodes[onode] += client.results[onode]
        
    # Save the outcome of the first qubit by default    
    outcomes_dict[circuit] = outcomes_sum_all_onodes[onodes[0]]
    outcome = int(outcomes_dict[circuit])
    majority_vote_outcome = "Ambig." if outcome == d/2 else int(outcome>d/2)
    p, expected_outcome = find_correct_value(circuit_name=circuit)
    print("#######")
    print(circuit)
    print(f"Prob. of getting 1: {p}")
    print(f"{outcome}/100 -> Noiseless outcome: {majority_vote_outcome}")
    print(f"Expected outcome of majority vote: {expected_outcome}")
