import numpy as np
import numpy.typing as npt
from graphix.sim.base_backend import State
from graphix.sim.density_matrix import DensityMatrix
from graphix.sim.statevec import Statevec


def fidelity(u: npt.NDArray[np.complex128], v: npt.NDArray[np.complex128]) -> float:
    return np.abs(np.dot(u.conjugate(), v))  # type: ignore[no-any-return]


def compare_backend_results(state1: State, state2: State) -> float:
    if isinstance(state1, Statevec) and isinstance(state2, Statevec):
        return fidelity(state1.flatten(), state2.flatten())
    if isinstance(state1, DensityMatrix):
        dm1 = state1
    elif isinstance(state1, Statevec):
        dm1 = DensityMatrix(state1)
    else:
        raise NotImplementedError
    if isinstance(state2, DensityMatrix):
        dm2 = state2
    elif isinstance(state2, Statevec):
        dm2 = DensityMatrix(state2)
    else:
        raise NotImplementedError
    return fidelity(dm1.rho.flatten(), dm2.rho.flatten())
