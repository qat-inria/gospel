"""Uncorrelated depolarising noise model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import typing_extensions
from graphix.channels import KrausChannel, KrausData
from graphix.command import BaseM, CommandKind
from graphix.noise_models.noise_model import (
    A,
    CommandOrNoise,
    Noise,
    NoiseCommands,
    NoiseModel,
)
from graphix.ops import Ops
from graphix.rng import ensure_rng

if TYPE_CHECKING:
    from numpy.random import Generator


@dataclass
class SinglePauliNoise(Noise):
    """One-qubit depolarising noise with probabibity `prob`."""

    prob: float
    error_type: Literal["X", "Z"] = "X"

    def nqubits(self) -> int:
        """Return the number of qubits targetted by the noise element."""
        return 1

    def to_kraus_channel(self) -> KrausChannel:
        """Return the Kraus channel describing the noise element."""
        if self.error_type == "Z":
            return KrausChannel([KrausData(self.prob, Ops.Z)])

        return KrausChannel([KrausData(self.prob, Ops.X)])


class SinglePauliNoiseModel(NoiseModel):
    """Test noise model on 3 qubit line graph and deterministic X or Z on middle qubit.

    edges: list of possible edges to draw from
    :param NoiseModel: Parent abstract class class:`graphix.noise_model.NoiseModel`
    :type NoiseModel: class
    """

    def __init__(
        self,
        prob: float,
        error_type: Literal["X", "Z"] = "X",
        rng: Generator | None = None,
    ) -> None:
        self.rng = ensure_rng(rng)
        self.prob = prob
        self.error_type = error_type

    def input_nodes(self, nodes: list[int]) -> NoiseCommands:
        """Return the noise to apply to input nodes."""
        return []

    def command(self, cmd: CommandOrNoise) -> NoiseCommands:
        """Return the noise to apply to the command `cmd`."""
        # flag to check of target node '1' has been visited to not apply noise twice
        if cmd.kind == CommandKind.N:
            return [cmd]
        if cmd.kind == CommandKind.E:
            if 0 in cmd.nodes and 1 in cmd.nodes:
                return [
                    cmd,
                    A(
                        noise=SinglePauliNoise(
                            self.prob, self.error_type
                        ),  # another thing where str is not subtype of Literal?
                        nodes=[1],
                    ),
                ]
            return [cmd]

        if cmd.kind == CommandKind.M:
            return [cmd]
        if cmd.kind == CommandKind.X:
            return [cmd]
        if cmd.kind == CommandKind.Z:
            return [cmd]
        # Use of `==` here for mypy
        if (
            cmd.kind == CommandKind.C  # noqa: PLR1714
            or cmd.kind == CommandKind.T
            or cmd.kind == CommandKind.A
            or cmd.kind == CommandKind.S
        ):
            return [cmd]
        typing_extensions.assert_never(cmd.kind)

    def confuse_result(self, cmd: BaseM, result: bool) -> bool:
        """Assign wrong measurement result cmd = "M"."""
        return result
