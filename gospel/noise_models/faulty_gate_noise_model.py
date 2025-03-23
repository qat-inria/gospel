"""Uncorrelated depolarising noise model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typing_extensions
from graphix.command import BaseM, CommandKind
from graphix.noise_models.noise_model import (
    A,
    CommandOrNoise,
    NoiseCommands,
    NoiseModel,
)
from graphix.rng import ensure_rng

if TYPE_CHECKING:
    from numpy.random import Generator


from graphix.noise_models.depolarising_noise_model import DepolarisingNoise


class FaultyCZNoiseModel(NoiseModel):
    """Uncorrelated depolarising noise model. with fixed gate.

    edges: list of possible edges to draw from
    :param NoiseModel: Parent abstract class class:`graphix.noise_model.NoiseModel`
    :type NoiseModel: class
    """

    def __init__(
        self,
        edges: frozenset[frozenset[int]],
        chosen_edges: frozenset[frozenset[int]] | None = None,
        edge_count: int | None = None,
        prepare_error_prob: float = 0.0,
        x_error_prob: float = 0.0,
        z_error_prob: float = 0.0,
        entanglement_error_prob: float = 0.0,
        measure_channel_prob: float = 0.0,
        measure_error_prob: float = 0.0,
        rng: Generator | None = None,
    ) -> None:
        self.edges = edges
        self.prepare_error_prob = prepare_error_prob
        self.x_error_prob = x_error_prob
        self.z_error_prob = z_error_prob
        self.entanglement_error_prob = entanglement_error_prob
        self.measure_error_prob = measure_error_prob
        self.measure_channel_prob = measure_channel_prob
        self.rng = ensure_rng(rng)

        # choose the target faulty gate
        # random for now
        # need the list type even for a single edge for the test
        # self.chosen_edges = [*self.rng.choice(list(self.edges), size=1).tolist()]

        print(f"chosen edges {chosen_edges}")

        if chosen_edges is not None:
            if edge_count is not None:
                raise ValueError(
                    "`chosen_edges` and `edge_count` are mutually exclusive"
                )
            self.chosen_edges = chosen_edges
        else:
            edge_list = list(self.edges)
            if edge_count is None:
                edge_count = min(6, len(edge_list))
            selected_indices = self.rng.choice(
                range(len(edge_list)), size=edge_count, replace=False
            )
            self.chosen_edges = frozenset(edge_list[i] for i in selected_indices)

            # list(self.edges)[[self.rng.integers(len(self.edges), size=6)] # [*self.rng.choice(list(self.edges), size=6).tolist()]

        print(f"chosen edges after {self.chosen_edges}")

    def input_nodes(self, nodes: list[int]) -> NoiseCommands:
        """Return the noise to apply to input nodes."""
        return [
            A(noise=DepolarisingNoise(self.prepare_error_prob), nodes=[node])
            for node in nodes
        ]

    def command(self, cmd: CommandOrNoise) -> NoiseCommands:
        """Return the noise to apply to the command `cmd`."""
        if cmd.kind == CommandKind.N:
            return [
                cmd,
                A(noise=DepolarisingNoise(self.prepare_error_prob), nodes=[cmd.node]),
            ]
        if cmd.kind == CommandKind.E:
            if (
                frozenset(cmd.nodes) in self.chosen_edges
            ):  # need symmetrisation since edges are directed
                u, v = cmd.nodes
                return [
                    cmd,
                    A(
                        noise=DepolarisingNoise(self.entanglement_error_prob),
                        nodes=[u],
                    ),
                    A(
                        noise=DepolarisingNoise(self.entanglement_error_prob),
                        nodes=[v],
                    ),
                ]
            return [cmd]

        if cmd.kind == CommandKind.M:
            return [
                A(noise=DepolarisingNoise(self.measure_channel_prob), nodes=[cmd.node]),
                cmd,
            ]
        if cmd.kind == CommandKind.X:
            return [
                cmd,
                A(noise=DepolarisingNoise(self.x_error_prob), nodes=[cmd.node]),
            ]
        if cmd.kind == CommandKind.Z:
            return [
                cmd,
                A(noise=DepolarisingNoise(self.z_error_prob), nodes=[cmd.node]),
            ]
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
        if self.rng.uniform() < self.measure_error_prob:
            return not result
        return result
