from pathlib import Path

from sampling_circuits.sample_circuits import sample_circuits

sample_circuits(
    ncircuits=10000,
    nqubits=5,
    depth=10,
    p_gate=0.5,
    p_cnot=0.5,
    p_cnot_flip=0.5,
    p_rx=0.5,
    seed=1729,
    threshold=0.4,
    target=Path("circuits/"),
)
