from __future__ import annotations

from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import pytest
from graphix import command
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
from gospel.noise_models.uncorrelated_depolarising_noise_model import (
    UncorrelatedDepolarisingNoiseModel,
)
from gospel.scripts.aces import Method, cli


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


@pytest.mark.parametrize("jumps", range(1, 2))
def test_single_deterministic_noisy_gate(fx_bg: PCG64, jumps: int) -> None:
    """test if ACES can find one faulty gate. Use the same noise model as hotgat.py.
    Use dask only locally."""

    freeze_support()
    cli(
        nqubits=3,
        nlayers=2,
        depol_prob=0.9,
        shots=1,
        ncircuits=10,
        verbose=False,
        method=Method.Stim,  # use Stim method
        scale=None,  # for local parallelism
        target=Path("plot.png"),
    )
