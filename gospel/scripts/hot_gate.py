import time
from pathlib import Path

import numpy as np
from graphix import command
from graphix.pattern import Pattern
from tqdm import tqdm
from veriphix.client import Client, Secrets
from veriphix.trappifiedCanvas import TrappifiedCanvas

from gospel.brickwork_state_transpiler import (
    ConstructionOrder,
    generate_random_pauli_pattern,
    get_bipartite_coloring,
)
from gospel.noise_models.faulty_gate_noise_model import FaultyCZNoiseModel
from gospel.scripts.qasm2brickwork_state import (
    draw_brickwork_state_colormap_from_pattern,
)
from gospel.stim_pauli_preprocessing import (
    StimBackend,
)


def perform_simulation(
    pattern: Pattern, depol_prob: float = 0.0, shots: int = 1
) -> list[dict[int, int]]:
    # NOTE data validation? nqubits, nlayers larger than 0, p between 0 and 1,n shots int >0

    # for order in (ConstructionOrder.Canonical, ConstructionOrder.Deviant):

    # dummy computation
    # only canonical ordering

    # Initialize secrets and client
    secrets = Secrets(r=False, a=False, theta=False)
    client = Client(pattern=pattern, secrets=secrets)

    # Get bipartite coloring and create test runs
    colours = get_bipartite_coloring(pattern)
    test_runs = client.create_test_runs(manual_colouring=colours)

    # Define noise model
    # don't reinitialise it since has its own randomness

    # noise_model = UncorrelatedDepolarisingNoiseModel(entanglement_error_prob=depol_prob)

    noise_model = FaultyCZNoiseModel(
        entanglement_error_prob=depol_prob, edges=set(pattern.get_graph()[1])
    )
    # noise_model = DepolarisingNoiseModel(entanglement_error_prob = 0.001)

    results_table = []
    n_failures = 0

    for i in tqdm(range(shots)):  # noqa: B007
        # reinitialise the backend!
        backend = StimBackend()
        # generate trappiefied canvas (input state is refreshed)

        run = TrappifiedCanvas(test_runs[rng.integers(len(test_runs))], rng=rng)

        # Delegate the test run to the client
        trap_outcomes = client.delegate_test_run(  # no noise model, things go wrong
            backend=backend, run=run, noise_model=noise_model
        )

        # Create a result dictionary (trap -> outcome)
        result = {
            tuple(trap): outcome for trap, outcome in zip(run.traps_list, trap_outcomes)
        }

        results_table.append(result)

        # Print pass/fail based on the sum of the trap outcomes
        if sum(trap_outcomes) != 0:
            n_failures += 1
            # print(f"Iteration {i}: ❌ Trap round failed", flush=True)
        else:
            pass
            # print(f"Iteration {i}: ✅ Trap round passed", flush=True)

    # Final report after completing the test rounds
    print(
        f"Final result: {n_failures}/{shots} failed rounds",
        flush=True,
    )
    print("-" * 50, flush=True)
    return results_table


def compute_failure_probabilities(
    results_table: list[dict[int, int]],
) -> dict[int, float]:
    occurences = {}
    occurences_one = {}

    for results in results_table:
        for q, r in results.items():
            if q not in occurences:
                occurences[q] = 1
                occurences_one[q] = r
            else:
                occurences[q] += 1
                if r == 1:
                    occurences_one[q] += 1

    return {q: occurences_one[q] / occurences[q] for q in occurences}


def plot_heatmap(data: dict[int, float], path: Path, target: str) -> None:
    draw_brickwork_state_colormap_from_pattern(
        pattern=pattern, target=path / target, failure_probas=data
    )


if __name__ == "__main__":
    # TODO do cli with typer!

    nqubits = 7
    nlayers = 2

    # initialising pattern
    rng = np.random.default_rng(12345)
    order = ConstructionOrder.Deviant  # ConstructionOrder.Deviant
    pattern = generate_random_pauli_pattern(
        nqubits=nqubits, nlayers=nlayers, order=order, rng=rng
    )
    # Add measurement commands to the output nodes
    for onode in pattern.output_nodes:
        pattern.add(command.M(node=onode))

    print("Starting simulation...")
    start = time.time()
    results_table = perform_simulation(pattern, depol_prob=0.5, shots=int(1e3))

    print(f"Simulation finished in {time.time() - start:.4f} seconds.")

    print("Computing failure probabilities...")
    failure_probas = compute_failure_probabilities(results_table)
    # print(f" final failure probas {failure_probas}")

    print("Plotting the heatmap...")

    # change this to save the figure
    directory_path = Path("simulation/results")
    # Create the directory
    directory_path.mkdir(parents=True, exist_ok=True)

    target = "hotgate_deviant.svg"
    plot_heatmap(failure_probas, directory_path, target)
    print("Done!")
