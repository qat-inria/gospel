from __future__ import annotations

import math
from multiprocessing import freeze_support
from time import time_ns

import dask.distributed
import numpy as np
import pytest
from graphix import Pattern, command
from graphix.command import CommandKind
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.simulator import DefaultMeasureMethod, FixedPrepareMethod
from numpy.random import PCG64, Generator
from veriphix.client import Client, Secrets, remove_flow
from veriphix.trappifiedCanvas import TrappifiedCanvas

from gospel.brickwork_state_transpiler import (
    ConstructionOrder,
    generate_random_pauli_pattern,
    get_bipartite_coloring,
)
from gospel.noise_models.faulty_gate_noise_model import FaultyCZNoiseModel
from gospel.noise_models.uncorrelated_depolarising_noise_model import (
    UncorrelatedDepolarisingNoiseModel,
)
from gospel.scripts.aces import (
    Method,
    compute_aces_postprocessing,
    generate_equations,
    perform_simulation,
)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_delegate_test_run(fx_bg: PCG64, jumps: int) -> None:
    nqubits = 3
    nlayers = 5
    depol_prob = 0.01
    order = ConstructionOrder.Canonical
    noise_model = UncorrelatedDepolarisingNoiseModel(entanglement_error_prob=depol_prob)
    rng = Generator(fx_bg.jumped(jumps))
    pattern = generate_random_pauli_pattern(
        nqubits=nqubits, nlayers=nlayers, order=order, rng=rng
    )

    # Add measurement commands to the output nodes
    for onode in pattern.output_nodes:
        pattern.add(command.M(node=onode))

    client_pattern = remove_flow(pattern)  # type: ignore[no-untyped-call]

    secrets = Secrets(r=False, a=False, theta=False)
    client = Client(pattern=pattern, secrets=secrets)

    # Get bipartite coloring and create test runs
    colours = get_bipartite_coloring(pattern)
    test_runs = client.create_test_runs(manual_colouring=colours)

    seed = rng.integers(2**32)
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    for i, col in enumerate(test_runs):
        # Define noise model

        # generate trappified canvas (input state is refreshed)

        backend = DensityMatrixBackend(rng=rng1)

        run = TrappifiedCanvas(col)
        trap_outcomes = client.delegate_test_run(
            backend=backend, run=run, noise_model=noise_model
        )
        results_veriphix = [
            {
                int(trap): outcome
                for (trap,), outcome in zip(run.traps_list, trap_outcomes)
            }
        ]
        measure_method = DefaultMeasureMethod()
        prepare_method = FixedPrepareMethod(dict(enumerate(run.states)))
        input_state = [run.states[i] for i in client_pattern.input_nodes]
        client_pattern.simulate_pattern(
            backend="densitymatrix",
            input_state=input_state,
            prepare_method=prepare_method,
            measure_method=measure_method,
            noise_model=noise_model,
            rng=rng2,
        )
        results_graphix = [
            {int(trap): measure_method.results[trap] for (trap,) in run.traps_list}
        ]
        assert results_veriphix == results_graphix


def generate_equations_reference(pattern: Pattern) -> dict[int, set[frozenset[int]]]:
    result = {}
    nodes, _edges = pattern.get_graph()
    for node in nodes:
        active_nodes = {node}
        lambda_set = set()
        for cmd in reversed(list(pattern)):
            if cmd.kind == CommandKind.E:
                u, v = cmd.nodes
                if u in active_nodes or v in active_nodes:
                    lambda_set.add(frozenset({u, v}))
                if u == node:
                    active_nodes.add(v)
                if v == node:
                    active_nodes.add(u)
        result[node] = lambda_set
    return result


def test_generate_equations() -> None:
    nqubits = 16
    nlayers = 16
    for order in list(ConstructionOrder):
        pattern = generate_random_pauli_pattern(nqubits, nlayers, order=order)
        assert generate_equations(pattern) == generate_equations_reference(pattern)


def test_benchmark_generate_equations() -> None:
    nqubits = 16
    nlayers = 16
    order = ConstructionOrder.Canonical
    pattern = generate_random_pauli_pattern(nqubits, nlayers, order=order)
    start = time_ns()
    _ = generate_equations(pattern)
    time = time_ns() - start
    start = time_ns()
    _ = generate_equations_reference(pattern)
    time_ref = time_ns() - start
    print(f"{time} / {time_ref}")


def test_one_brick() -> None:
    pattern = generate_random_pauli_pattern(2, 1, order=ConstructionOrder.Canonical)
    eqns = generate_equations(pattern)
    edges = list(pattern.edges)
    m = [[int(edge in s) for edge in edges] for _node, s in eqns.items()]
    print(f"{len(m)=}")
    print(f"{np.linalg.matrix_rank(m)=}")


@pytest.mark.parametrize("jumps", range(1, 2))
def test_single_deterministic_noisy_gate(fx_bg: PCG64, jumps: int) -> None:
    """test if ACES can find one faulty gate. Use the same noise model as hotgat.py.
    Use dask only locally."""

    # define noise model
    # choose first edge.
    # nqubits = 3
    # chosen_edges = frozenset([frozenset((nqubits, 2 * nqubits))])

    # noise_model = FaultyCZNoiseModel(
    #     entanglement_error_prob=params.depol_prob,
    #     edges=pattern.edges,
    #     chosen_edges=chosen_edges,
    # )

    # add noise model in params
    # pattern doesn't exist outside cli...

    # Test passed for Stim, Veriphix (with old stim implem) and Graphix!!
    freeze_support()

    nqubits = 4
    nlayers = 3
    depol_prob = 0.2
    nshots = 10000
    ncircuits = 1
    method = Method.Stim

    cluster = dask.distributed.LocalCluster()
    dask_client = dask.distributed.Client(cluster)  # type: ignore[no-untyped-call]

    # TODO change to n_nodes
    node = nqubits * ((4 * nlayers) + 1)

    pattern = generate_random_pauli_pattern(
        nqubits, nlayers, order=ConstructionOrder.Deviant
    )
    print(pattern)
    _nodes, edges_list = pattern.get_graph()

    problematic_edges = set()

    for u, v in edges_list:
        chosen_edges = frozenset([frozenset((u, v))])
        # chosen_edges = frozenset([frozenset((nqubits, 2 * nqubits))])

        noise_model = FaultyCZNoiseModel(
            entanglement_error_prob=depol_prob,
            chosen_edges=chosen_edges,
        )
        # noise_model = UncorrelatedDepolarisingNoiseModel(
        #     entanglement_error_prob=params.depol_prob
        # )

        # print(f"checking depol param {params.depol_prob}")

        results = perform_simulation(
            nqubits=nqubits,
            nlayers=nlayers,
            noise_model=noise_model,
            nshots=nshots,
            ncircuits=ncircuits,
            method=method,
            dask_client=dask_client,
        )

        x = compute_aces_postprocessing(nqubits, node, nlayers, results)

        try:
            detected_edge = None
            for i, v2 in enumerate(x):
                if math.isclose(v2, 1 - depol_prob * 4 / 3, abs_tol=0.05):
                    if detected_edge is None:
                        detected_edge = i
                    else:
                        raise ValueError("Already detected edge")
                else:
                    assert math.isclose(v2, 1, abs_tol=0.05)
            if detected_edge is None:
                raise ValueError("No detected edge")
        except (ValueError, AssertionError):
            problematic_edges.add((u, v))

    if problematic_edges:
        raise ValueError(f"{problematic_edges=}")
