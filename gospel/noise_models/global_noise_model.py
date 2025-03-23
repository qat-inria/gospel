from __future__ import annotations

from typing import TYPE_CHECKING

from graphix.noise_models.noise_model import (
    CommandOrNoise,
    NoiseCommands,
    NoiseModel,
)
from graphix.rng import ensure_rng

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graphix.command import BaseM
    from numpy.random import Generator


class GlobalNoiseModel(NoiseModel):
    """Global noise model.

    :param NoiseModel: Parent abstract class class:`graphix.noise_model.NoiseModel`
    :type NoiseModel: class
    """

    def __init__(
        self,
        nodes: Iterable[int],
        prob: float = 0.0,
        rng: Generator | None = None,
    ) -> None:
        self.prob = prob
        self.nodes = list(nodes)
        self.rng = ensure_rng(rng)
        self.refresh_randomness()

    def refresh_randomness(self) -> None:
        self.node = self.nodes[self.rng.integers(len(self.nodes))]

    def input_nodes(self, nodes: list[int]) -> NoiseCommands:
        """Return the noise to apply to input nodes."""
        return []

    def command(self, cmd: CommandOrNoise) -> NoiseCommands:
        """Return the noise to apply to the command `cmd`."""
        return [cmd]

    def confuse_result(self, cmd: BaseM, result: bool) -> bool:
        """Assign wrong measurement result cmd = "M"."""
        if cmd.node == self.node and self.rng.uniform() < self.prob:
            return not result
        return result
