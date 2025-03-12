from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING

import dask.distributed
import graphix.command
from dask_jobqueue import SLURMCluster
from graphix.noise_models import NoiseModel
from graphix.rng import ensure_rng
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.states import BasicStates
from veriphix.client import Client, Secrets, TrappifiedCanvas

import gospel.brickwork_state_transpiler
from gospel.scripts.qasm2brickwork_state import read_qasm

if TYPE_CHECKING:
    from graphix.command import BaseM
    from graphix.noise_models.noise_model import (
        CommandOrNoise,
        NoiseCommands,
    )

## Load a circuit with success probability p = 0.7839549798834848
# BQP error
# context handler open renvoie f et à la fin ferme le fichier
# valeur à durer de vie, resource libéré.

with Path("circuits/circuit000.qasm").open() as f:
    circuit = read_qasm(f)

print(circuit.instruction)

pattern = gospel.brickwork_state_transpiler.transpile(circuit)

print(list(pattern))


## Measure output nodes, to have classical output
classical_output = pattern.output_nodes
for onode in classical_output:
    pattern.add(graphix.command.M(node=onode))

states = [BasicStates.PLUS] * len(pattern.input_nodes)

# correct since the pattern is transpiled from a circuit and hence has a causal flow
pattern.minimize_space()

print(f"Number of nodes in the pattern : {pattern.n_node}")


def load_pattern_from_circuit(circuit_label: str):
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


pattern, onodes = load_pattern_from_circuit("circuit000.qasm")
print(onodes)

with Path("circuits/table.json").open() as f:
    table = json.load(f)
    circuits = [name for name, prob in table.items() if prob < 0.1]
    print(len(circuits))

"""Global noise model."""


if TYPE_CHECKING:
    from numpy.random import Generator


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

    def confuse_result(self, cmd: BaseM, result: bool) -> bool:
        """Assign wrong measurement result cmd = "M"."""
        if cmd.node == self.node and self.rng.uniform() < self.prob:
            return not result
        return result


threshold, p_err = 0.2, 0.6
threshold, p_err = 0.1, 0.6
threshold, p_err = 0.2, 0.6
threshold, p_err = 0.1, 0.1

# Recording info
fail_rates = []
decision_dict = {}
outcomes_dict = {}

# Fixed parameters
d = 20  # nr of computation rounds
t = 10  # nr of test rounds
N = d + t  # nr of total rounds
num_instances = 10
instances = random.sample(circuits, num_instances)


backend = DensityMatrixBackend()

portdash = 10000 + os.getuid()
cluster = SLURMCluster(
    account="inria",
    queue="cpu_devel",
    cores=1,
    memory="1GB",
    walltime="00:01:00",
    scheduler_options={"dashboard_address": f":{portdash}"},
)
cluster.scale(10)
dask_client = dask.distributed.Client(cluster)


def for_each_instance(circuit):
    # Generate a different instance
    pattern, onodes = load_pattern_from_circuit(circuit)

    # Instanciate Client and create Test runs
    client = Client(pattern=pattern, secrets=Secrets(a=True, r=True, theta=True))
    colours = gospel.brickwork_state_transpiler.get_bipartite_coloring(pattern)
    test_runs = client.create_test_runs(manual_colouring=colours)

    outcome_sum = 0
    # Trappified scheme parameters

    rounds = list(range(N))
    random.shuffle(rounds)

    n_failed_trap_rounds = 0
    n_tolerated_failures = threshold * t

    noise_model = GlobalNoiseModel(prob=p_err, nodes=range(pattern.n_node))

    def for_each_round(i):
        if i < d:
            # Computation round
            client.delegate_pattern(backend=backend, noise_model=noise_model)
            return ("computation", client.results[onodes[0]])
        # Test round
        run = TrappifiedCanvas(random.choice(test_runs))
        trap_outcomes = client.delegate_test_run(
            run=run, backend=backend, noise_model=noise_model
        )
        noise_model.refresh_randomness()

        # Record trap failure
        # A trap round fails if one of the single-qubit traps failed
        return ("test", sum(trap_outcomes) != 0)

    outcome = dask_client.gather(dask_client.map(for_each_round, rounds))
    outcome_sum = sum(value for kind, value in outcome if kind == "computation")
    n_failed_trap_rounds = sum(value for kind, value in outcome if kind == "test")

    if n_failed_trap_rounds > n_tolerated_failures:
        # reject instance
        # do nothing
        return None
    # accept instance
    # compute majority vote
    # if outcome_sum == d/2:
    #    raise ValueError("Ambiguous result")
    return int(outcome_sum > d / 2)


outcome = dask_client.gather(dask_client.map(for_each_instance, instances))

outcomes_dict = dict(zip(instances, outcome))

print(outcomes_dict)
