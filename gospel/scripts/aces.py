from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from multiprocessing import freeze_support
from pathlib import Path
from typing import TYPE_CHECKING

import dask.distributed
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
import typer
from graphix import Pattern, command
from graphix.command import CommandKind
from graphix.sim.statevec import Statevec
from graphix.simulator import DefaultMeasureMethod, PrepareMethod
from graphix.states import BasicState, State
from numpy.random import PCG64, Generator
from veriphix.client import Client, Secrets, remove_flow
from veriphix.trappifiedCanvas import TrappifiedCanvas

from gospel.brickwork_state_transpiler import (
    ConstructionOrder,
    generate_random_pauli_pattern,
    get_bipartite_coloring,
)
from gospel.cluster.dask_interface import get_cluster
from gospel.noise_models.single_pauli_noise_model import (
    SinglePauliNoiseModel,  # noqa: F401
)
from gospel.noise_models.uncorrelated_depolarising_noise_model import (
    UncorrelatedDepolarisingNoiseModel,
)
from gospel.stim_pauli_preprocessing import StimBackend, pattern_to_stim_circuit

if TYPE_CHECKING:
    from graphix.command import BaseN
    from graphix.noise_models.noise_model import NoiseModel
    from graphix.sim.base_backend import Backend

logger = logging.getLogger(__name__)


class Method(Enum):
    Stim = "stim"
    Graphix = "graphix"
    Veriphix = "veriphix"


def state_to_basic_state(state: State) -> BasicState:
    bs = BasicState.try_from_statevector(Statevec(state).psi)
    if bs is None:
        raise ValueError(f"Not a basic state: {state}")
    return bs


@dataclass
class SingleSimulation:
    order: ConstructionOrder
    nqubits: int
    nlayers: int
    noise_model: NoiseModel
    nshots: int
    jumps: int
    method: Method


@dataclass
class FixedPrepareMethod(PrepareMethod):
    states: dict[int, State]

    def prepare(self, backend: Backend, cmd: BaseN) -> None:
        backend.add_nodes(nodes=[cmd.node], data=self.states[cmd.node])


def perform_single_simulation(
    params: SingleSimulation,
) -> list[
    tuple[
        ConstructionOrder,
        bool,
        list[tuple[list[list[int]], list[int], list[list[int]]]],
    ]
]:
    fx_bg = PCG64(42)

    rng = Generator(fx_bg.jumped(params.jumps))  # Use the jumped rng

    pattern = generate_random_pauli_pattern(
        nqubits=params.nqubits, nlayers=params.nlayers, order=params.order, rng=rng
    )
    # Add measurement commands to the output nodes
    for onode in pattern.output_nodes:
        pattern.add(command.M(node=onode))

    noise_model = params.noise_model

    client_pattern = remove_flow(pattern)  # type: ignore[no-untyped-call]

    secrets = Secrets(r=False, a=False, theta=False)
    client = Client(pattern=pattern, secrets=secrets)

    # Get bipartite coloring and create test runs
    colours = get_bipartite_coloring(pattern)
    test_runs = client.create_test_runs(manual_colouring=colours)

    outcomes = []

    for i, col in enumerate(test_runs):
        run = TrappifiedCanvas(col)
        if params.method == Method.Veriphix:
            assert params.nshots == 1
            backend = StimBackend()
            trap_outcomes = client.delegate_test_run(  # no noise model, things go wrong
                backend=backend, run=run, noise_model=noise_model
            )
            results = [
                ([trap_outcomes], list(range(len(trap_outcomes))), run.traps_list)
            ]
        elif params.method == Method.Graphix:
            assert params.nshots == 1
            measure_method = DefaultMeasureMethod()
            prepare_method = FixedPrepareMethod(dict(enumerate(run.states)))
            input_state = [run.states[i] for i in client_pattern.input_nodes]
            client_pattern.simulate_pattern(
                backend="densitymatrix",
                input_state=input_state,
                prepare_method=prepare_method,
                measure_method=measure_method,
                noise_model=noise_model,
            )
            results = [
                (
                    [measure_method.results],
                    list(range(len(measure_method.results))),
                    run.traps_list,
                )
            ]
        else:
            input_state_dict = {}
            fixed_states = {}
            for node, state in enumerate(run.states):
                basic_state = state_to_basic_state(state)
                if node in client_pattern.input_nodes:
                    input_state_dict[node] = basic_state
                else:
                    fixed_states[node] = basic_state
            circuit, measure_indices = pattern_to_stim_circuit(
                client_pattern,
                input_state=input_state_dict,
                noise_model=noise_model,
                fixed_states=fixed_states,
            )
            sample = circuit.compile_sampler().sample(shots=params.nshots)
            results = [(sample, measure_indices, run.traps_list)]

        # Choose the correct outcome table based on order
        outcomes.append((params.order, bool(i), results))

    return outcomes


@dataclass
class SimulationResult:
    canonical: list[tuple[list[list[int]], list[int], list[list[int]]]]
    deviant: list[tuple[list[list[int]], list[int], list[list[int]]]]


def perform_simulation(
    nqubits: int,
    nlayers: int,
    noise_model: NoiseModel,
    nshots: int,
    ncircuits: int,
    method: Method,
    dask_client: dask.distributed.Client | None,
) -> SimulationResult:
    jobs = [
        SingleSimulation(
            order=order,
            nqubits=nqubits,
            nlayers=nlayers,
            noise_model=noise_model,
            nshots=nshots,
            method=method,
            jumps=circuit * 2 + int(order == ConstructionOrder.Deviant),
        )
        for circuit in range(ncircuits)
        for order in (ConstructionOrder.Canonical, ConstructionOrder.Deviant)
    ]

    logger.debug(f"nb jobs to run: {len(jobs)}")
    if dask_client is None:
        outcomes = list(map(perform_single_simulation, jobs))
    else:
        outcomes = dask_client.gather(dask_client.map(perform_single_simulation, jobs))  # type: ignore[no-untyped-call]

    test_outcome_table_canonical = []
    test_outcome_table_deviant = []

    for outcome in outcomes:
        for order, _col, results in outcome:
            if order == ConstructionOrder.Canonical:
                test_outcome_table_canonical.extend(results)
            elif order == ConstructionOrder.Deviant:
                test_outcome_table_deviant.extend(results)

    return SimulationResult(test_outcome_table_canonical, test_outcome_table_deviant)


def compute_failure_probabilities(
    nnodes: int,
    results_table: list[tuple[list[list[int]], list[int], list[list[int]]]],
) -> npt.NDArray[np.float64]:
    occurrences = np.zeros(nnodes, dtype=np.int64)
    occurrences_one = np.zeros(nnodes, dtype=np.int64)

    for samples, measure_indices, traps_list in results_table:
        nsamples = len(samples)
        ones = np.array(samples).sum(axis=0)
        for (trap,) in traps_list:
            occurrences[trap] += nsamples
            occurrences_one[trap] += ones[measure_indices[trap]]

    return occurrences_one / occurrences


def generate_equations(pattern: Pattern) -> dict[int, set[frozenset[int]]]:
    nodes = pattern.nodes
    result = {node: set() for node in nodes}
    active_nodes = {node: {node} for node in nodes}
    for cmd in reversed(list(pattern)):
        if cmd.kind == CommandKind.E:
            u, v = cmd.nodes
            edge = frozenset({u, v})
            for target in active_nodes[u] | active_nodes[v]:
                result[target].add(edge)
            active_nodes[u].add(v)
            active_nodes[v].add(u)
    return result


@dataclass
class EdgeDependency:
    edge: frozenset[int]
    order: ConstructionOrder
    measure_index: int
    previous_edges: frozenset[int]


def generate_edge_dependencies(nqubits: int, nlayers: int) -> list[EdgeDependency]:
    equations = {}
    for order in (ConstructionOrder.Canonical, ConstructionOrder.Deviant):
        pattern = generate_random_pauli_pattern(nqubits, nlayers, order=order)
        equations[order] = generate_equations(pattern)
    result = []
    known = set()
    known_indices = {}
    while True:
        new_element = False
        for order in (ConstructionOrder.Canonical, ConstructionOrder.Deviant):
            for measure_index, lambdas in equations[order].items():
                left = lambdas - known
                try:
                    (edge,) = left
                except ValueError:
                    pass
                else:
                    previous_edges = frozenset(
                        {known_indices[lam] for lam in lambdas if lam != edge}
                    )
                    dependency = EdgeDependency(
                        edge, order, measure_index, previous_edges
                    )
                    known.add(edge)
                    known_indices[edge] = len(result)
                    result.append(dependency)
                    new_element = True
        if not new_element:
            break
    return result


def compute_aces_postprocessing_iteratively(
    nnodes: int, dependencies: list[EdgeDependency], results: SimulationResult
) -> dict[frozenset[int], float]:
    start = time.time()
    pi = {
        ConstructionOrder.Canonical: compute_failure_probabilities(
            nnodes, results.canonical
        ),
        ConstructionOrder.Deviant: compute_failure_probabilities(
            nnodes, results.deviant
        ),
    }
    logger.info(f"Failure probabilities in {time.time() - start:.4f} seconds.")
    result_log = []
    for dependency in dependencies:
        pi_value = math.log(1 - 2 * pi[dependency.order][dependency.measure_index])
        for edge in dependency.previous_edges:
            pi_value -= result_log[edge]
        result_log.append(pi_value)
    return {
        dependency.edge: math.exp(v) for dependency, v in zip(dependencies, result_log)
    }


def compute_probabilities_difference_can(
    failure_proba_can_result: dict[int, float],
    n_nodes: int,
) -> list[float]:
    # return [1 - 2 * v for _, v  in sorted(failure_proba_can_result.items(), key=lambda x: x[0])]
    return [1 - 2 * failure_proba_can_result[k] for k in range(n_nodes)]


def compute_probabilities_difference_dev(
    failure_proba_dev_result: dict[int, float],
    n_nodes: int,
    n_qubits: int,
) -> list[float]:
    return [
        1 - 2 * failure_proba_dev_result[k]
        for k in range(n_nodes)
        if (k % n_qubits) % 2 == 0 and (k // n_qubits) % 2 == 1
    ]


def generate_qubit_edge_matrix_from_pattern(
    pattern: Pattern, nodes: list[int], edges: list[frozenset[int]]
) -> npt.NDArray[np.int64]:
    equations = generate_equations(pattern)
    edge_index = {edge: index for index, edge in enumerate(edges)}
    matrix = np.zeros((len(nodes), len(edges)), dtype=np.int64)
    for row, node in enumerate(nodes):
        lambdas = equations[node]
        for edge in lambdas:
            matrix[row, edge_index[edge]] = 1
    return matrix


def generate_qubit_edge_matrix(
    nqubits: int, nlayers: int
) -> tuple[list[frozenset[int]], npt.NDArray[np.int64]]:
    pattern_can = generate_random_pauli_pattern(
        nqubits=nqubits, nlayers=nlayers, order=ConstructionOrder.Canonical
    )
    pattern_dev = generate_random_pauli_pattern(
        nqubits=nqubits, nlayers=nlayers, order=ConstructionOrder.Deviant
    )

    edges_set = pattern_can.edges
    assert pattern_dev.edges == edges_set
    edges = list(edges_set)

    nnodes = nqubits * (4 * nlayers + 1)
    nodes_can = list(range(nnodes))
    nodes_dev = [
        row + col * nqubits
        for col in range(1, 4 * nlayers + 1, 2)
        for row in range(0, nqubits, 2)
    ]

    qubit_edge_matrix = generate_qubit_edge_matrix_from_pattern(
        pattern_can, nodes_can, edges
    )
    qubit_edge_matrix_dev = generate_qubit_edge_matrix_from_pattern(
        pattern_dev, nodes_dev, edges
    )

    # Stack the matrices together to form a single system
    lhs = np.vstack(
        (qubit_edge_matrix, qubit_edge_matrix_dev)
    )  # Combine coefficient matrices
    logger.debug(f"{lhs.shape=}")
    logger.debug(f"{lhs=}")
    return edges, lhs


def compute_aces_postprocessing(
    nqubits: int, nnodes: int, nlayers: int, results: SimulationResult
) -> dict[frozenset[int], float]:
    logger.info("Computing failure probabilities...")
    failure_proba_can_final = compute_failure_probabilities(nnodes, results.canonical)
    failure_proba_dev_all = compute_failure_probabilities(nnodes, results.deviant)

    # computing circuit eigenvalues for both orders
    # deviant has been filtered to remove redundancy
    failure_proba_can = compute_probabilities_difference_can(
        failure_proba_can_final, nnodes
    )
    failure_proba_dev = compute_probabilities_difference_dev(
        failure_proba_dev_all,
        nnodes,
        nqubits,
    )

    logger.debug(f"failure proba canonical {failure_proba_can}")
    logger.debug(f"failure proba deviant {failure_proba_dev}")

    # convert to numpy arrays for later processing
    py_failure_proba_can = np.array(failure_proba_can, dtype=np.float64)
    logger.debug(py_failure_proba_can.shape)
    py_failure_proba_dev = np.array(failure_proba_dev, dtype=np.float64)
    logger.debug(py_failure_proba_dev.shape)

    logger.info("Setting up ACES...")
    edges, lhs = generate_qubit_edge_matrix(nqubits, nlayers)
    rhs = np.concatenate(
        (py_failure_proba_can, py_failure_proba_dev)
    )  # Combine constant vectors
    logger.debug(f"{rhs.shape=}")
    logger.debug(f"{rhs=}")

    log_rhs = np.log(rhs)  # log constant vectors

    log_params, *_ = np.linalg.lstsq(lhs, log_rhs, rcond=None)

    logger.debug(f"log {log_params}")

    logger.info("Calculating the lambdas...")
    return dict(
        zip(edges, np.exp(log_params))
    )  # Convert log values back to original variables


def generate_plot(
    inferred_lambdas: list[float],
    vline: float,
    target: Path = Path("plot.png"),
) -> None:
    """generate plot based on data: absolute value.
    vline for position of theoretical expectation
    """

    plt.figure(figsize=(10, 6))

    # Create histogram with density curve
    plt.hist(
        inferred_lambdas,
        bins="auto",
        color="#2ecc71",
        edgecolor="#27ae60",
        alpha=0.7,
        # range=(0, 1),
        density=True,
    )

    # Add KDE plot
    sns.kdeplot(  # type: ignore[no-untyped-call]
        inferred_lambdas,
        color="#34495e",
        linewidth=2,
        label="KDE",
    )

    # Add reference lines
    plt.axvline(
        vline, color="red", linestyle="--", linewidth=1.5
    )  # , label="(λ(diff)=0.0)""
    plt.axvline(
        np.mean(inferred_lambdas),
        color="#3498db",
        linestyle="-",
        linewidth=1.5,
        label=f"Mean ({np.mean(inferred_lambdas):.2f})",
    )
    plt.axvline(
        np.median(inferred_lambdas),
        color="#9b59b6",
        linestyle="-",
        linewidth=1.5,
        label=f"Median ({np.median(inferred_lambdas):.2f})",
    )

    # Formatting
    plt.title(r"ACES", fontsize=14)
    if vline != 0:
        plt.xlabel(r"${\hat \lambda}$", fontsize=12)
    else:
        plt.xlabel(r"${\hat \lambda} - \lambda_{\rm th}$", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    sns.despine()  # type: ignore[no-untyped-call]

    # Add statistical annotations
    stats_text = (
        f"Total edges: {len(inferred_lambdas)}\n"
        f"Min: {np.min(inferred_lambdas):.3f}\n"
        f"Max: {np.max(inferred_lambdas):.3f}\n"
        f"Std: {np.std(inferred_lambdas):.3f}"
    )
    plt.text(
        0.75,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.9},
    )

    plt.tight_layout()
    # plt.show()
    plt.savefig(target)


def cli(
    nqubits: int = 5,
    nlayers: int = 10,
    depol_prob: float = 0.001,
    nshots: int = 10000,
    ncircuits: int = 1,
    verbose: bool = False,
    method: Method | None = None,
    walltime: int | None = None,
    memory: int | None = None,
    cores: int | None = None,
    port: int | None = None,
    scale: int | None = None,
    target: Path = Path("plot.png"),
) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.setLevel(level)

    if walltime is None and scale is None:
        dask_client = None
    else:
        cluster = get_cluster(walltime, memory, cores, port, scale)
        dask_client = dask.distributed.Client(cluster)  # type: ignore[no-untyped-call]

    nnodes = nqubits * ((4 * nlayers) + 1)

    # typer seems not to support default values for Enum
    if method is None:
        method = Method.Stim

    # choose first edge.
    # chosen_edges = frozenset([frozenset((0, nqubits))])
    # logger.debug(f"Chosen edges {chosen_edges}")

    # chosen_edges = frozenset([frozenset((nqubits, 2 * nqubits))])

    # noise_model = FaultyCZNoiseModel(
    #     entanglement_error_prob=depol_prob,
    #     chosen_edges=chosen_edges,
    # )
    noise_model = UncorrelatedDepolarisingNoiseModel(entanglement_error_prob=depol_prob)

    # print(f"checking depol param {params.depol_prob}")

    logger.info("Starting simulations...")
    start = time.time()
    results = perform_simulation(
        nqubits=nqubits,
        nlayers=nlayers,
        noise_model=noise_model,
        nshots=nshots,
        ncircuits=ncircuits,
        method=method,
        dask_client=dask_client,
    )

    logger.info(f"Simulation finished in {time.time() - start:.4f} seconds.")
    start = time.time()
    dependencies = generate_edge_dependencies(nqubits, nlayers)
    inferred_lambdas = compute_aces_postprocessing_iteratively(
        nnodes, dependencies, results
    ).values()
    # inferred_lambdas = compute_aces_postprocessing(
    #    nqubits, nnodes, nlayers, results
    # ).values()

    logger.info(f"Lambda inferred in {time.time() - start:.4f} seconds.")
    logger.debug(f"Inferred lambdas {inferred_lambdas}")

    # expected theoretical value of the lambdas
    lambda_expected = 1 - 4 / 3 * depol_prob
    # compute difference between inferred and theoretical values
    lambda_diff: list[float] = [l - lambda_expected for l in inferred_lambdas]

    logger.info("Plotting the result...")

    # absolute plot
    generate_plot(
        list(inferred_lambdas), vline=1 - 4 * depol_prob / 3
    )  # typing to list necessary?

    # difference wrt theoretical expectation
    generate_plot(
        list(lambda_diff), vline=0, target=Path("plot_diff.png")
    )  # typing to list necessary?
    logger.info("Done!")


if __name__ == "__main__":
    freeze_support()
    typer.run(cli)
