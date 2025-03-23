from __future__ import annotations

import enum
import json
import random
import socket
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

import dask.distributed
import typer
from dask_jobqueue import SLURMCluster
from graphix import command
from graphix.noise_models import NoiseModel
from graphix.rng import ensure_rng
from graphix.sim.density_matrix import DensityMatrixBackend
from veriphix.client import Client, Secrets
from veriphix.trappifiedCanvas import TrappifiedCanvas, TrapStabilizers

import gospel.brickwork_state_transpiler
from gospel.scripts.qasm_parser import read_qasm

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graphix import Pattern
    from graphix.command import BaseM
    from graphix.noise_models.noise_model import (
        CommandOrNoise,
        NoiseCommands,
    )


def load_pattern_from_circuit(circuit_label: str) -> tuple[Pattern, list[int]]:
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
    circuits = [name for name, prob in table.items() if prob < 0.4]
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


@dataclass
class Parameters:
    d: int
    t: int
    N: int
    num_instances: int
    threshold: float
    p_err: float


@dataclass
class Rounds:
    parameters: Parameters
    circuit_name: str
    client: Client
    onodes: list[int]
    test_runs: list[TrapStabilizers]
    rounds: list[int]


def get_rounds(parameters: Parameters, circuit_name: str) -> Rounds:
    # Generate a different instance
    pattern, onodes = load_pattern_from_circuit(circuit_name)

    # Instanciate Client and create Test runs
    client = Client(pattern=pattern, secrets=Secrets(a=True, r=True, theta=True))
    colours = gospel.brickwork_state_transpiler.get_bipartite_coloring(pattern)
    test_runs = client.create_test_runs(manual_colouring=colours)

    rounds = list(range(parameters.N))
    random.shuffle(rounds)

    return Rounds(parameters, circuit_name, client, onodes, test_runs, rounds)


class RoundKind(Enum):
    Computation = enum.auto()
    Test = enum.auto()


@dataclass
class RoundResult:
    kind: RoundKind
    value: bool


RoundResultOrException = RoundResult | Exception


@dataclass
class ComputationResult:
    hostname: str
    i: int
    round_result: RoundResultOrException


def for_each_round(
    args: tuple[Rounds, int],
) -> ComputationResult:
    rounds, i = args
    try:
        noise_model = GlobalNoiseModel(
            prob=rounds.parameters.p_err,
            nodes=range(rounds.client.initial_pattern.n_node),
        )
        backend = DensityMatrixBackend()

        if i < rounds.parameters.d:
            # Computation round
            rounds.client.delegate_pattern(backend=backend, noise_model=noise_model)
            result: RoundResultOrException = RoundResult(
                RoundKind.Computation, bool(rounds.client.results[rounds.onodes[0]])
            )
        else:
            # Test round
            run = TrappifiedCanvas(random.choice(rounds.test_runs))
            trap_outcomes = rounds.client.delegate_test_run(
                run=run, backend=backend, noise_model=noise_model
            )
            noise_model.refresh_randomness()

            # Record trap failure
            # A trap round fails if one of the single-qubit traps failed
            result = RoundResult(RoundKind.Test, bool(sum(trap_outcomes) != 0))
    except Exception as e:
        result = e
    return ComputationResult(socket.gethostname(), i, result)


def for_all_rounds(rounds: Rounds) -> tuple[str, list[ComputationResult]]:
    return rounds.circuit_name, [for_each_round((rounds, i)) for i in rounds.rounds]


def run(
    d: int,
    t: int,
    num_instances: int,
    threshold: float,
    p_err: float,
    walltime: int | None = None,
    memory: int | None = None,
    cores: int | None = None,
    port: int | None = None,
    scale: int | None = None,
) -> None:
    if walltime is None and memory is None and cores is None and port is None:
        cluster = dask.distributed.LocalCluster()
    else:
        if walltime is None:
            raise ValueError("--walltime <hours> is required for running on cleps")
        if memory is None:
            raise ValueError("--memory <GB> is required for running on cleps")
        if cores is None:
            raise ValueError("--cores <N> is required for running on cleps")
        if port is None:
            raise ValueError("--port <N> is required for running on cleps")
        if scale is None:
            raise ValueError("--scale <N> is required for running on cleps")
        cluster = SLURMCluster(
            account="inria",
            queue="cpu_devel",
            cores=cores,
            memory=f"{memory}GB",
            walltime=f"{walltime}:00:00",
            scheduler_options={"dashboard_address": f":{port}"},
        )
    if scale is not None:
        cluster.scale(scale)

    parameters = Parameters(
        d=d, t=t, N=d + t, num_instances=num_instances, threshold=threshold, p_err=p_err
    )

    # Recording info
    circuit_names = random.sample(circuits, parameters.num_instances)

    all_rounds = [
        get_rounds(parameters, circuit_name) for circuit_name in circuit_names
    ]

    n_failed_trap_rounds = 0
    # n_tolerated_failures = parameters.threshold * parameters.t

    dask_client = dask.distributed.Client(cluster)
    outcome_circuits = dict(
        dask_client.gather(
            dask_client.map(
                for_all_rounds,
                all_rounds,
            )
        )
    )

    with open(f"w{parameters.threshold}-p{p_err}-raw.json", "w") as file:
        file.write(str(outcome_circuits))

    outcomes_dict = {}

    for circuit_name, results in outcome_circuits.items():
        outcome_sum = 0
        n_failed_trap_rounds = 0
        for computation_result in results:
            result = computation_result.round_result
            if isinstance(result, Exception):
                print(result)
            elif result.kind == RoundKind.Computation:
                outcome_sum += result.value
            elif result.kind == RoundKind.Test:
                n_failed_trap_rounds += result.value
            else:
                assert_never(result.kind)
        failure_rate = n_failed_trap_rounds / parameters.t
        decision = (
            failure_rate > parameters.threshold
        )  # True if the instance is accepted, False if rejected
        if outcome_sum == parameters.d / 2:
            outcome: str | int = "Ambig."
        else:
            outcome = int(outcome_sum > parameters.d / 2)
        outcomes_dict[circuit_name] = {
            "outcome_sum": outcome_sum,
            "n_failed_trap_rounds": n_failed_trap_rounds,
            "decision": decision,
            "outcome": outcome,
            "failure_rate": failure_rate,
        }

    with open(f"w{parameters.threshold}-p{p_err}.json", "w") as file:
        json.dump(outcomes_dict, file, indent=4)


if __name__ == "__main__":
    typer.run(run)
