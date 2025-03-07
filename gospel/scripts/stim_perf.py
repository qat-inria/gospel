import stim
from graphix.noise_models.depolarising_noise_model import DepolarisingNoiseModel

from gospel.brickwork_state_transpiler import generate_random_pauli_pattern
from gospel.stim_pauli_preprocessing import (
    simulate_pauli,
)


def perform_random_depolarising_simulation(
    nqubits: int, nlayers: int, p: float = 0
) -> dict[int, bool]:
    # NOTE data validation? nqubits, nlayers larger than 0, p between 0 and 1.

    # generate random pauli brickwork
    nqubits = 2
    nlayers = 2
    # default order is canonical so don't manipulate the pattern afterwards (standardisation, ...)!
    pattern = generate_random_pauli_pattern(nqubits=nqubits, nlayers=nlayers)
    # all qubits are already measured
    sim = stim.TableauSimulator()
    noise_model = DepolarisingNoiseModel(entanglement_error_prob=p)
    return simulate_pauli(sim, pattern, noise_model)


if __name__ == "__main__":
    res = perform_random_depolarising_simulation(nqubits=2, nlayers=2)
    print(res)
