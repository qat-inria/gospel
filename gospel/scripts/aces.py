import time

from graphix import command
from numpy.random import PCG64, Generator
from tqdm import tqdm
from veriphix.client import Client, Secrets
from veriphix.trappifiedCanvas import TrappifiedCanvas

from gospel.brickwork_state_transpiler import (
    ConstructionOrder,
    generate_random_pauli_pattern,
    get_bipartite_coloring,
)
from gospel.noise_models.uncorrelated_depolarising_noise_model import (
    UncorrelatedDepolarisingNoiseModel,
)
from gospel.stim_pauli_preprocessing import (
    StimBackend,
)


def perform_simulation(
    nqubits: int, nlayers: int, depol_prob: float = 0.0, shots: int = 1
) -> tuple[list[dict[int, int]], list[dict[int, int]]]:
    # Initialization
    fx_bg = PCG64(42)
    jumps = 5
    # Number of test iterations

    # Define separate outcome tables for Canonical and Deviant
    test_outcome_table_canonical: list[dict[int, int]] = []
    test_outcome_table_deviant: list[dict[int, int]] = []

    # Loop over Construction Orders
    for order in (ConstructionOrder.Canonical, ConstructionOrder.Deviant):
        rng = Generator(fx_bg.jumped(jumps))  # Use the jumped rng

        # TODO not really needed
        # just two patterns are enough...
        pattern = generate_random_pauli_pattern(
            nqubits=nqubits, nlayers=nlayers, order=order, rng=rng
        )

        # Add measurement commands to the output nodes
        for onode in pattern.output_nodes:
            pattern.add(command.M(node=onode))

        secrets = Secrets(r=False, a=False, theta=False)
        client = Client(pattern=pattern, secrets=secrets)

        # Get bipartite coloring and create test runs
        colours = get_bipartite_coloring(pattern)
        test_runs = client.create_test_runs(manual_colouring=colours)

        # Define noise model
        noise_model = UncorrelatedDepolarisingNoiseModel(
            entanglement_error_prob=depol_prob
        )

        n_failures = 0

        # Choose the correct outcome table based on order
        if order == ConstructionOrder.Canonical:
            test_outcome_table = test_outcome_table_canonical
        else:
            test_outcome_table = test_outcome_table_deviant

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
                tuple(trap): outcome
                for trap, outcome in zip(run.traps_list, trap_outcomes)
            }

            test_outcome_table.append(result)

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
    return test_outcome_table_canonical, test_outcome_table_deviant


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


if __name__ == "__main__":
    # TODO do cli with typer!

    nqubits = 7
    nlayers = 2

    print("Starting simulations...")
    start = time.time()
    results_canonical, results_deviant = perform_simulation(
        nqubits=nqubits, nlayers=nlayers, depol_prob=0.5, shots=int(1e3)
    )

    print(f"Simulation finished in {time.time() - start:.4f} seconds.")

    print("Computing failure probabilities...")
    failure_probas_can = compute_failure_probabilities(results_canonical)
    failure_probas_dev = compute_failure_probabilities(results_deviant)

    print("Setting up ACES...")

    print("Done!")
