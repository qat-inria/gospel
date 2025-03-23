import pytest
from graphix import command
from numpy.random import Generator

from gospel.brickwork_state_transpiler import (
    ConstructionOrder,
    generate_random_pauli_pattern,
)
from gospel.scripts.hot_gate import compute_failure_probabilities, perform_simulation


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
