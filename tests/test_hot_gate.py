import pytest
from graphix import command
from numpy.random import Generator

from gospel.brickwork_state_transpiler import (
    ConstructionOrder,
    generate_random_pauli_pattern,
    get_hot_traps_of_faulty_gate,
)
from gospel.scripts.hot_gate import (
    CHOSEN_EDGES,
    Method,
    compute_failure_probabilities,
    perform_simulation,
)


@pytest.mark.parametrize("order", list(ConstructionOrder))
def test_compute_failure_probabilities(
    fx_rng: Generator, order: ConstructionOrder
) -> None:
    nqubits = 7
    nlayers = 2
    shots = 1000
    depol_prob = 0.5
    pattern = generate_random_pauli_pattern(
        nqubits=nqubits, nlayers=nlayers, order=order, rng=fx_rng
    )
    for onode in pattern.output_nodes:
        pattern.add(command.M(node=onode))
    results_table = perform_simulation(pattern, depol_prob=depol_prob, shots=shots)
    compute_failure_probabilities(results_table)


@pytest.mark.parametrize("order", list(ConstructionOrder))
@pytest.mark.parametrize(
    "method",
    [Method.StimBackend],
)
def test_hot_gates(
    fx_rng: Generator,
    order: ConstructionOrder,
    method: Method,
) -> None:
    nqubits = 7
    nlayers = 2
    shots = 1000
    depol_prob = 0.5
    threshold = 0.2
    pattern = generate_random_pauli_pattern(
        nqubits=nqubits, nlayers=nlayers, order=order, rng=fx_rng
    )
    for onode in pattern.output_nodes:
        pattern.add(command.M(node=onode))
    results_table = perform_simulation(
        pattern,
        method=method,
        depol_prob=depol_prob,
        shots=shots,
        chosen_edges=CHOSEN_EDGES,
    )
    failure_probas = compute_failure_probabilities(results_table)
    hot_traps = {trap for trap, proba in failure_probas.items() if proba >= threshold}
    expected_hot_traps = {
        trap
        for u, v in CHOSEN_EDGES
        for trap in get_hot_traps_of_faulty_gate(nqubits, order, (u, v))[1]
    }
    assert hot_traps == expected_hot_traps
