import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
import typer
from graphix import Pattern, command
from graphix.sim.statevec import Statevec
from graphix.states import BasicState, State
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
from gospel.stim_pauli_preprocessing import pattern_to_stim_circuit


def state_to_basic_state(state: State) -> BasicState:
    bs = BasicState.try_from_statevector(Statevec(state).psi)
    if bs is None:
        raise ValueError(f"Not a basic state: {state}")
    return bs


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
    test_outcome_table: dict[tuple[ConstructionOrder, bool], list[dict[int, int]]] = {}

    noise_model = UncorrelatedDepolarisingNoiseModel(entanglement_error_prob=depol_prob)
    # Loop over Construction Orders
    for order in tqdm((ConstructionOrder.Canonical, ConstructionOrder.Deviant)):
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

        for i, col in enumerate(test_runs):
            # Define noise model

            # generate trappified canvas (input state is refreshed)

            run = TrappifiedCanvas(col)

            # all nodes have to be prepared for test runs
            # don't reinitialise them since we don't care for blindness right now
            input_state = {
                i: state_to_basic_state(state) for i, state in enumerate(run.states)
            }

            print(f"len input ste {len(input_state)}")
            print(f"input ste {input_state}")

            client_pattern = Pattern(input_nodes=pattern.input_nodes)
            for cmd in pattern:
                if isinstance(cmd, command.M):
                    client_pattern.add(command.M(node=cmd.node))
                else:
                    client_pattern.add(cmd)

            circuit, measure_indices = pattern_to_stim_circuit(
                client_pattern,
                input_state=input_state,
                noise_model=noise_model,
            )

            sample = circuit.compile_sampler().sample(shots=shots // 2)

            # Choose the correct outcome table based on order

            test_outcome_table[(order, bool(i))] = [
                {trap: s[measure_indices[trap]] for (trap,) in run.traps_list}
                for s in sample
            ]

    test_outcome_table_canonical = (
        test_outcome_table[(ConstructionOrder.Canonical, False)]
        + test_outcome_table[(ConstructionOrder.Canonical, True)]
    )
    separate_binary_lists_can = [[0 if v == np.False_ else 1 for v in d.values()] for d in test_outcome_table_canonical]
    n_failures = 0
    total_rows = len(separate_binary_lists_can)
    print("For canonical order:")
    for row_idx, row in enumerate(separate_binary_lists_can):
        row_sum = sum(row)
        if row_sum != 0:
            n_failures += 1
            #print(f"Iteration {row_idx}: ❌ Trap round failed", flush=True)
        else:
            pass
            #print(f"Iteration {row_idx}: ✅ Trap round passed", flush=True)

    print(
            f"Final result: {n_failures}/{total_rows} failed rounds",
            flush=True,
        )
    
    test_outcome_table_deviant = (
        test_outcome_table[(ConstructionOrder.Deviant, False)]
        + test_outcome_table[(ConstructionOrder.Deviant, True)]
    )
    separate_binary_lists_dev = [[0 if v == np.False_ else 1 for v in d.values()] for d in test_outcome_table_deviant]
    n_failures_dev = 0
    total_rows_dev = len(separate_binary_lists_can)
    print("-" * 50, flush=True)
    print("For deviant order:")
    for row_idx, row in enumerate(separate_binary_lists_dev):
        row_sum_dev = sum(row)
        if row_sum_dev != 0:
            n_failures_dev += 1
            #print(f"Iteration {row_idx}: ❌ Trap round failed", flush=True)
        else:
            pass
            #print(f"Iteration {row_idx}: ✅ Trap round passed", flush=True)

    print(
            f"Final result: {n_failures_dev}/{total_rows_dev} failed rounds",
            flush=True,
        )
    
    #print(f"LEN {test_outcome_table_canonical}")
    # convert to old format
    # Create a result dictionary (trap -> outcome)
    # result = {
    #     trap: outcome for (trap,), outcome in zip(run.traps_list, trap_outcomes)
    # }

    # test_outcome_table.append(result)

    # Print pass/fail based on the sum of the trap outcomes
    # if sum(trap_outcomes) != 0:
    #     n_failures += 1
    #     # print(f"Iteration {i}: ❌ Trap round failed", flush=True)
    # else:
    #     pass
    #     # print(f"Iteration {i}: ✅ Trap round passed", flush=True)

    # # Final report after completing the test rounds
    # print(
    #     f"Final result: {n_failures}/{shots} failed rounds",
    #     flush=True,
    # )
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


def compute_failure_probabilities_can(
    failure_proba_can_result: dict[int, float],
) -> list[float]:
    failure_proba_can_array = [
        v for k, v in sorted(failure_proba_can_result.items(), key=lambda x: x[0])
    ]
    failure_proba_can_inverted = [1 - x for x in failure_proba_can_array]
    return [
        abs(orig - inv)
        for orig, inv in zip(failure_proba_can_array, failure_proba_can_inverted)
    ]


def compute_failure_probabilities_dev(
    failure_proba_dev_result: dict[int, float], n_qubits: int, max_index: int
) -> list[float]:
    required_indices = []
    start = n_qubits
    max_index = max_index - 1

    while start <= max_index:
        for offset in range(0, n_qubits, 2):
            current_index = start + offset
            if current_index > max_index:
                break
            required_indices.append(current_index)  # Note the comma to create tuple
        start += 2 * n_qubits

    failure_proba_dev_final = {
        idx: failure_proba_dev_result[idx]
        for idx in required_indices
        if idx in failure_proba_dev_result
    }

    failure_proba_dev_array = [
        v for k, v in sorted(failure_proba_dev_final.items(), key=lambda x: x[0])
    ]
    failure_proba_dev_inverted = [1 - x for x in failure_proba_dev_array]
    return [
        abs(origi - inve)
        for origi, inve in zip(failure_proba_dev_array, failure_proba_dev_inverted)
    ]


Coords2D = tuple[int, int]
Edge = tuple[Coords2D, Coords2D]
MatrixAndMaps = tuple[
    npt.NDArray[np.int64],
    dict[Coords2D, int],
    dict[Edge, int],
]
Conditions = list[Callable[[int, int], tuple[bool, list[Edge]]]]


def generate_qubit_edge_matrix_with_unknowns_can(
    n_qubits: int, n_layers: int
) -> MatrixAndMaps:
    n = n_qubits
    m = 4 * n_layers + 1
    qubits = {}  # Mapping from (i, j) to qubit index
    edges = {}  # Mapping from edge (start, end) to edge index
    edge_index = 0
    qubit_index = 0

    # Assign an index to each qubit (i, j)
    for j in range(m):
        for i in range(n):
            qubits[(i, j)] = qubit_index
            qubit_index += 1

    # Collect all edges and assign them an index
    for i in range(n):
        for j in range(m - 1):
            edge = ((i, j), (i, j + 1))  # Horizontal edge
            edges[edge] = edge_index
            edge_index += 1

    for i in range(n - 1):
        for j in range(m):
            if (
                (j + 1) % 8 == 3 and (i + 1) % 2 != 0 and j + 3 < m
            ):  # Column j ≡ 3 (mod 8) and odd row i
                edge = ((i, j), (i + 1, j))
                edges[edge] = edge_index
                edge_index += 1
                edge = ((i, j + 2), (i + 1, j + 2))
                edges[edge] = edge_index
                edge_index += 1
            if (
                (j + 1) % 8 == 7 and (i + 1) % 2 == 0 and j + 3 < m
            ):  # Column j ≡ 7 (mod 8) and even row i
                edge = ((i, j), (i + 1, j))
                edges[edge] = edge_index
                edge_index += 1
                edge = ((i, j + 2), (i + 1, j + 2))
                edges[edge] = edge_index
                edge_index += 1

    # Create the symbolic matrix (qubits × edges)
    matrix = np.zeros((len(qubits), len(edges)), dtype=object)

    # Create symbolic variables for edges
    # edge_symbols = [sp.symbols(f'x{i}') for i in range(len(edges))]

    # Apply special conditions for qubits
    conditions: Conditions = [
        (
            lambda i, j: (
                (i % 2 == 0 and (j % 8 == 0 or j % 8 == 6))
                or (i % 2 == 1 and (j % 8 == 2 or j % 8 == 4)),
                [
                    ((i, j - 2), (i, j - 1)),
                    ((i, j - 1), (i, j)),
                    ((i - 1, j - 1), (i - 1, j)),
                    ((i - 1, j), (i, j)),
                    ((i, j), (i, j + 1)),
                ],
            )
        ),
        (
            lambda i, j: (
                (i % 2 == 1 and (j % 8 == 0 or j % 8 == 6))
                or (i % 2 == 0 and (j % 8 == 2 or j % 8 == 4)),
                [
                    ((i, j - 2), (i, j - 1)),
                    ((i, j - 1), (i, j)),
                    ((i + 1, j - 1), (i + 1, j)),
                    ((i, j), (i + 1, j)),
                    ((i, j), (i, j + 1)),
                ],
            )
        ),
        (
            lambda i, j: (
                (i % 2 == 0 and (j % 8 == 1 or j % 8 == 7))
                or (i % 2 == 1 and (j % 8 == 3 or j % 8 == 5)),
                [
                    ((i, j - 2), (i, j - 1)),
                    ((i - 1, j - 1), (i, j - 1)),
                    ((i, j - 1), (i, j)),
                    ((i, j), (i, j + 1)),
                ],
            )
        ),
        (
            lambda i, j: (
                (i % 2 == 1 and (j % 8 == 1 or j % 8 == 7))
                or (i % 2 == 0 and (j % 8 == 3 or j % 8 == 5)),
                [
                    ((i, j - 2), (i, j - 1)),
                    ((i, j - 1), (i + 1, j)),
                    ((i, j - 1), (i, j)),
                    ((i, j), (i, j + 1)),
                ],
            )
        ),
    ]

    for f in conditions:
        for i in range(n):
            for j in range(m):
                condition, special_edges = f(i, j)
                if condition:
                    for (rel_i1, rel_j1), (rel_i2, rel_j2) in special_edges:
                        # Compute actual coordinates of edge
                        edge = ((rel_i1, rel_j1), (rel_i2, rel_j2))

                        # Ensure the edge exists before modifying the matrix
                        if edge in edges:
                            e_idx = edges[edge]
                            q_idx = qubits[(i, j)]
                            # Use symbolic variables for edges
                            # matrix[q_idx, e_idx] = edge_symbols[e_idx]
                            matrix[q_idx, e_idx] = 1

    # Return matrix with symbolic edge variables
    return matrix, qubits, edges


def generate_qubit_edge_matrix_with_unknowns_dev(
    noqubits: int, nolayers: int
) -> MatrixAndMaps:
    # assert n % 2 == 0, "The number of rows (n) must be even."
    n = noqubits
    m = 4 * nolayers + 1
    qubits_dev = {}  # Mapping from (i, j) to qubit index
    edges_dev = {}  # Mapping from edge (start, end) to edge index
    edge_index_dev = 0
    qubit_index_dev = 0

    # Assign an index to each qubit (i, j)
    for j in range(m):
        for i in range(n):
            if i % 2 == 0 and (j % 8 == 1 or j % 8 == 3 or j % 8 == 5 or j % 8 == 7):
                qubits_dev[(i, j)] = qubit_index_dev
                qubit_index_dev += 1

    # Collect all edges and assign them an index
    for i in range(n):
        for j in range(m - 1):
            edge_dev = ((i, j), (i, j + 1))  # Horizontal edge
            edges_dev[edge_dev] = edge_index_dev
            edge_index_dev += 1

    for i in range(n - 1):
        for j in range(m):
            if (
                (j + 1) % 8 == 3 and (i + 1) % 2 != 0 and j + 3 < m
            ):  # Column j ≡ 3 (mod 8) and odd row i
                edge_dev = ((i, j), (i + 1, j))
                edges_dev[edge_dev] = edge_index_dev
                edge_index_dev += 1
                edge_dev = ((i, j + 2), (i + 1, j + 2))
                edges_dev[edge_dev] = edge_index_dev
                edge_index_dev += 1
            if (
                (j + 1) % 8 == 7 and (i + 1) % 2 == 0 and j + 3 < m
            ):  # Column j ≡ 7 (mod 8) and even row i
                edge_dev = ((i, j), (i + 1, j))
                edges_dev[edge_dev] = edge_index_dev
                edge_index_dev += 1
                edge_dev = ((i, j + 2), (i + 1, j + 2))
                edges_dev[edge_dev] = edge_index_dev
                edge_index_dev += 1

    # Create the symbolic matrix (qubits × edges)
    matrix_dev = np.zeros((len(qubits_dev), len(edges_dev)), dtype=object)

    # Create symbolic variables for edges
    # edge_symbols_dev = [sp.symbols(f'x{i}') for i in range(len(edges_dev))]

    # Apply special conditions for qubits
    conditions_dev: Conditions = [
        (
            lambda i, j: (
                (i % 2 == 0 and j % 8 == 1),
                [
                    ((i, j - 2), (i, j - 1)),
                    ((i - 1, j - 1), (i, j - 1)),
                    ((i, j - 1), (i, j)),
                    ((i, j), (i, j + 1)),
                    ((i, j + 1), (i + 1, j + 1)),
                ],
            )
        ),
        (
            lambda i, j: (
                (i % 2 == 0 and j % 8 == 3),
                [
                    ((i, j - 2), (i, j - 1)),
                    ((i, j - 1), (i + 1, j - 1)),
                    ((i, j - 1), (i, j)),
                    ((i, j), (i, j + 1)),
                    ((i, j + 1), (i + 1, j + 1)),
                ],
            )
        ),
        (
            lambda i, j: (
                (i % 2 == 0 and j % 8 == 5),
                [
                    ((i, j - 2), (i, j - 1)),
                    ((i, j - 1), (i + 1, j - 1)),
                    ((i, j - 1), (i, j)),
                    ((i, j), (i, j + 1)),
                    ((i - 1, j + 1), (i, j + 1)),
                ],
            )
        ),
        (
            lambda i, j: (
                (i % 2 == 0 and j % 8 == 7),
                [
                    ((i, j - 2), (i, j - 1)),
                    ((i - 1, j - 1), (i, j - 1)),
                    ((i, j - 1), (i, j)),
                    ((i, j), (i, j + 1)),
                    ((i - 1, j + 1), (i, j + 1)),
                ],
            )
        ),
    ]

    # print(edges_dev)
    for f in conditions_dev:
        for i in range(n):
            for j in range(m):
                if i % 2 == 0 and (
                    j % 8 == 1 or j % 8 == 3 or j % 8 == 5 or j % 8 == 7
                ):
                    condition_dev, special_edges_dev = f(i, j)
                    if condition_dev:
                        for (rel_i1, rel_j1), (rel_i2, rel_j2) in special_edges_dev:
                            # Compute actual coordinates of edge
                            edge_dev = ((rel_i1, rel_j1), (rel_i2, rel_j2))

                            # Ensure the edge exists before modifying the matrix
                            if edge_dev in edges_dev:
                                e_idx_dev = edges_dev[edge_dev]
                                q_idx_dev = qubits_dev[(i, j)]
                                # Use symbolic variables for edges
                                # matrix_dev[q_idx_dev, e_idx_dev] = edge_symbols_dev[e_idx_dev]
                                matrix_dev[q_idx_dev, e_idx_dev] = 1
    # Return matrix with symbolic edge variables
    return matrix_dev, qubits_dev, edges_dev


def cli(
    nqubits: int = 5, nlayers: int = 10, depol_prob: float = 0.001, shots: int = 10
) -> None:
    node = nqubits * ((4 * nlayers) + 1)

    print("Starting simulations...")
    start = time.time()
    results_canonical, results_deviant = perform_simulation(
        nqubits=nqubits, nlayers=nlayers, depol_prob=depol_prob, shots=shots
    )

    print(f"Simulation finished in {time.time() - start:.4f} seconds.")

    print("Computing failure probabilities...")
    failure_proba_can_final = compute_failure_probabilities(results_canonical)
    failure_proba_dev_all = compute_failure_probabilities(results_deviant)

    failure_proba_can = compute_failure_probabilities_can(failure_proba_can_final)
    failure_proba_dev = compute_failure_probabilities_dev(
        failure_proba_dev_all, nqubits, node
    )

    print(f"failute proba canonical {failure_proba_can}")
    print(f"failute proba deviant {failure_proba_dev}")
    py_failure_proba_can = np.array(failure_proba_can, dtype=np.float64)
    print(py_failure_proba_can.shape)
    py_failure_proba_dev = np.array(failure_proba_dev, dtype=np.float64)
    print(py_failure_proba_dev.shape)

    print("Setting up ACES...")
    qubit_edge_matrix, qubit_map, edge_map = (
        generate_qubit_edge_matrix_with_unknowns_can(nqubits, nlayers)
    )
    qubit_edge_matrix_dev, qubit_map_dev, edge_map_dev = (
        generate_qubit_edge_matrix_with_unknowns_dev(nqubits, nlayers)
    )
    qubit_edge_matrix = np.array(qubit_edge_matrix, dtype=np.float64)
    qubit_edge_matrix_dev = np.array(qubit_edge_matrix_dev, dtype=np.float64)
    print(qubit_edge_matrix.shape)
    print(qubit_edge_matrix_dev.shape)

    # Stack the matrices together to form a single system
    lhs = np.vstack(
        (qubit_edge_matrix, qubit_edge_matrix_dev)
    )  # Combine coefficient matrices
    rhs = np.concatenate(
        (py_failure_proba_can, py_failure_proba_dev)
    )  # Combine constant vectors
    print(lhs.shape, lhs)
    print(rhs.shape, rhs)

    log_rhs = np.log(rhs)  # log constant vectors

    log_params, *_ = np.linalg.lstsq(lhs, log_rhs, rcond=None)

    print(f"log {log_params}")

    print("Calculating the lambdas...")
    x = np.exp(log_params)  # Convert log values back to original variables
    lamba_initial = 1 - (4/3) * depol_prob
    x_diff = [(dif - lamba_initial) for dif in x]

    print(f"X {x}")
    print("Plotting the result...")

    plt.figure(figsize=(10, 6))

    # Create histogram with density curve
    n, bins, patches = plt.hist(
        x_diff,
        bins="auto",
        color="#2ecc71",
        edgecolor="#27ae60",
        alpha=0.7,
        density=True,
    )

    # Add KDE plot
    sns.kdeplot(  # type: ignore[no-untyped-call]
        x_diff,
        color="#34495e",
        linewidth=2,
        label=r"Density of $\lambda(diff)_{\mathrm{edge}}$",
    )

    # Add reference lines
    plt.axvline(0.0, color="red", linestyle="--", linewidth=1.5, label="(λ(diff)=0.0)")
    plt.axvline(
        np.mean(x_diff),
        color="#3498db",
        linestyle="-",
        linewidth=1.5,
        label=f"Mean ({np.mean(x_diff):.2f})",
    )
    plt.axvline(
        np.median(x_diff),
        color="#9b59b6",
        linestyle="-",
        linewidth=1.5,
        label=f"Median ({np.median(x_diff):.2f})",
    )

    # Formatting
    plt.title(r"ACES", fontsize=14)
    plt.xlabel(r"$\lambda(diff)_{\mathrm{edge}}$", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    sns.despine()  # type: ignore[no-untyped-call]

    # Add statistical annotations
    stats_text = (
        f"Total edges: {len(x_diff)}\n"
        f"Min: {np.min(x_diff):.2f}\n"
        f"Max: {np.max(x_diff):.2f}\n"
        f"Std: {np.std(x_diff):.2f}"
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
    plt.savefig("plot.png")
    print("Done!")


if __name__ == "__main__":
    typer.run(cli)
