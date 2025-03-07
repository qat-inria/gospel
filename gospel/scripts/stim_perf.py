import time

import matplotlib.pyplot as plt
import stim
from graphix.noise_models.depolarising_noise_model import DepolarisingNoiseModel

from gospel.brickwork_state_transpiler import generate_random_pauli_pattern
from gospel.stim_pauli_preprocessing import (
    simulate_pauli,
)


def perform_random_depolarising_simulation(
    nqubits: int, nlayers: int, depol_prob: float = 0, shots: int = 1
) -> float:  # dict[int, bool]
    # NOTE data validation? nqubits, nlayers larger than 0, p between 0 and 1,n shots int >0

    # generate random pauli brickwork
    # default order is canonical so don't manipulate the pattern afterwards (standardisation, ...)!
    pattern = generate_random_pauli_pattern(nqubits=nqubits, nlayers=nlayers)
    # print(f"nodes {pattern.get_graph()[0]}")
    # all qubits are already measured
    sim = stim.TableauSimulator()

    # MODIFY THE NOISE MODEL HERE
    noise_model = DepolarisingNoiseModel(entanglement_error_prob=depol_prob)

    start = time.time()
    for _ in range(shots):
        # print(simulate_pauli(sim, pattern, noise_model))
        simulate_pauli(sim, pattern, noise_model)
    duration = time.time() - start
    return duration  # noqa: RET504


def generate_benchmark_data(
    nqubits: int, max_depth: int, depol_prob: float, shots: int = 100
) -> list[float]:
    """generate benchmark data for a fixed depolarising probability, number of shots and qubits but varying brick depth

    Parameters
    ----------
    nqubits : int
        _description_
    max_depth : int
        BRICK depth
    depol_prob : float
        _description_
    shots : int, optional
        _description_, by default 100
    """
    return [
        perform_random_depolarising_simulation(
            nqubits=nqubits, nlayers=depth, shots=shots, depol_prob=depol_prob
        )
        for depth in range(1, max_depth + 1)
    ]


def plot_data(
    nqubits: int, max_depth: int, depol_prob: float, shots: int = 100
) -> None:
    """Plot data for a given number of qubits, up to brick depth `max_depth`and a fixed number of shots.
    The plot will compare the runtime of noiseless simulations and noisy simulation (depolarising noise) of strength `depol_prob`.

    Parameters
    ----------
    nqubits : int
        _description_
    max_depth : int
        _description_
    depol_prob : float
        _description_
    shots : int, optional
        _description_, by default 100
    """
    data = generate_benchmark_data(
        nqubits=nqubits, max_depth=max_depth, depol_prob=0, shots=shots
    )
    noisy_data = generate_benchmark_data(
        nqubits=nqubits, max_depth=max_depth, depol_prob=depol_prob, shots=shots
    )
    plt.plot(range(1, max_depth + 1), data, "+", ms=10, label="Noiseless")
    plt.plot(
        range(1, max_depth + 1),
        noisy_data,
        "o",
        markerfacecolor="none",
        ms=10,
        label=rf"Noisy, $p=${depol_prob}",
    )  # , markeredgecolor='r'
    plt.xlabel("brick depth")
    plt.ylabel("time (s)")
    plt.title(rf"Runtime for {nqubits} qubits and {shots} shots")
    plt.legend()
    plt.xticks(
        ticks=range(1, max_depth + 1), labels=[str(i) for i in range(1, max_depth + 1)]
    )
    # plt.show()


if __name__ == "__main__":
    # print(f"time {perform_random_depolarising_simulation(
    #     nqubits=2, nlayers=4, shots=3, depol_prob=0.0
    # )} s")

    print(generate_benchmark_data(nqubits=2, max_depth=4, depol_prob=0, shots=100))

    plot_data(nqubits=2, max_depth=4, depol_prob=0.75, shots=int(1e3))

    # res = perform_random_depolarising_simulation(nqubits=2, nlayers=2)
    # print(res)
