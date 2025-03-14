from __future__ import annotations

import itertools
import json
import os
import random
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import dask.distributed
from dask_jobqueue import SLURMCluster
from graphix import command
from graphix.noise_models import NoiseModel
from graphix.rng import ensure_rng
from graphix.sim.density_matrix import DensityMatrixBackend
from veriphix.client import Client, Secrets, TrappifiedCanvas, TrapStabilizers

import gospel.brickwork_state_transpiler
from gospel.cluster.run_on_cleps import run_on_cleps
from gospel.scripts.qasm2brickwork_state import read_qasm

if TYPE_CHECKING:
    from graphix.command import BaseM
    from graphix.noise_models.noise_model import (
        CommandOrNoise,
        NoiseCommands,
    )


def load_pattern_from_circuit(circuit_label: str):
    with Path(f"circuits/{circuit_label}").open() as f:
        circuit = read_qasm(f)
        pattern = gospel.brickwork_state_transpiler.transpile(circuit)

        ## Measure output nodes, to have classical output
        classical_output = pattern.output_nodes
        for onode in classical_output:
            pattern.add(command.M(node=onode))

        # states = [BasicStates.PLUS] * len(pattern.input_nodes)

        # correct since the pattern is transpiled from a circuit and hence has a causal flow
        pattern.minimize_space()
    return pattern, classical_output


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


# Fixed parameters
d = 100  # nr of computation rounds
t = 100  # nr of test rounds
N = d + t  # nr of total rounds
num_instances = 100


threshold, p_err = 0.2, 0.6
threshold, p_err = 0.1, 0.6
threshold, p_err = 0.2, 0.6
threshold, p_err = 0.1, 0.1


@dataclass
class Rounds:
    circuit_name: str
    client: Client
    onodes: list[int]
    test_runs: list[TrapStabilizers]
    rounds: list[int]


def get_rounds(circuit_name: str) -> Rounds:
    # Generate a different instance
    pattern, onodes = load_pattern_from_circuit(circuit_name)

    # Instanciate Client and create Test runs
    client = Client(pattern=pattern, secrets=Secrets(a=True, r=True, theta=True))
    colours = gospel.brickwork_state_transpiler.get_bipartite_coloring(pattern)
    test_runs = client.create_test_runs(manual_colouring=colours)

    rounds = list(range(N))
    random.shuffle(rounds)

    return Rounds(circuit_name, client, onodes, test_runs, rounds)


def for_each_round(args):
    rounds, i = args
    noise_model = GlobalNoiseModel(
        prob=p_err, nodes=range(rounds.client.initial_pattern.n_node)
    )
    backend = DensityMatrixBackend()

    if i < d:
        # Computation round
        rounds.client.delegate_pattern(backend=backend, noise_model=noise_model)
        result = ("computation", rounds.client.results[rounds.onodes[0]])
    else:
        # Test round
        run = TrappifiedCanvas(random.choice(rounds.test_runs))
        trap_outcomes = rounds.client.delegate_test_run(
            run=run, backend=backend, noise_model=noise_model
        )
        noise_model.refresh_randomness()

        # Record trap failure
        # A trap round fails if one of the single-qubit traps failed
        result = ("test", sum(trap_outcomes) != 0)
    return (rounds.circuit_name, (socket.gethostname(), i, result))


def run() -> None:
    if run_on_cleps:
        portdash = 10000 + os.getuid()
        cluster = SLURMCluster(
            account="inria",
            queue="cpu_devel",
            cores=8,
            memory="1GB",
            walltime="00:30:00",
            scheduler_options={"dashboard_address": f":{portdash}"},
        )
        cluster.scale(20)
    else:
        cluster = dask.distributed.LocalCluster()
        cluster.scale(5)

    # Recording info
    circuit_names = random.sample(circuits, num_instances)

    all_rounds = [get_rounds(circuit_name) for circuit_name in circuit_names]

    n_failed_trap_rounds = 0
    n_tolerated_failures = threshold * t

    dask_client = dask.distributed.Client(cluster)
    outcome = list(
        dask_client.gather(
            dask_client.map(
                for_each_round,
                [(rounds, i) for rounds in all_rounds for i in rounds.rounds],
            )
        )
    )
    print("here")

    outcome_circuits = dict(itertools.groupby(outcome, lambda pair: pair[0]))

    outcomes_dict = {}

    for circuit_name, results in outcome_circuits.items():
        outcome_sum = sum(
            value for _h, _i, (kind, value) in results if kind == "computation"
        )
        n_failed_trap_rounds = sum(
            value for _h, _i, (kind, value) in results if kind == "test"
        )

        if n_failed_trap_rounds > n_tolerated_failures:
            # reject instance
            # do nothing
            return
        # accept instance
        # compute majority vote
        # if outcome_sum == d/2:
        #    raise ValueError("Ambiguous result")
        outcomes_dict[circuit_name] = int(outcome_sum > d / 2)

    print(outcomes_dict)


if __name__ == "__main__":
    run()
