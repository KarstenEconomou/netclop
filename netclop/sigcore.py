"""Defines classes for significance clustering."""

import dataclasses
import typing
from collections import namedtuple

import click
import numpy as np

from .constants import Node, Partition, DEFAULT_SC_CONFIG

Score = namedtuple("Score", ["size", "pen"])

@dataclasses.dataclass
class SigClu:
    """Finds significant core of nodes within a module."""
    partition: set[Node]
    bootstraps: tuple[Partition, ...]

    config: dict[str, any] = dataclasses.field(default_factory=lambda: DEFAULT_SC_CONFIG)
    rng: np.random.Generator = np.random.default_rng(42)  # TODO: use config seed

    def find_sig_core(self, module: set[Node]) -> set[Node]:
        """Finds significant core of a module."""
        click.echo(f"Finding significant core of {len(module)}-node module")
        # Handle trivial modules
        num_nodes = len(module)
        match num_nodes:
            case 0:
                return set()
            case 1:
                return module
            case _:
                pass
        module = list(module)

        # Initialize state
        state = self.initialize_state(module)
        score = self.score(state, num_nodes)
        temp = self.config["temp_init"]

        # Core loop
        for i in range(self.config["iter_max"]):
            did_accept = False
            for _ in range(num_nodes):
                # Flip one random node's membership from candidate state and score
                node = self.rng.choice(module)
                new_state = self.flip(state, node)
                new_score = self.score(new_state, num_nodes)

                # Query accepting perturbed state
                if self.do_accept_state(score, new_score, temp):
                    state = new_state
                    score = new_score
                    did_accept = True

            if not did_accept:
                break
            click.echo(f"ITER: {i}, TEMP: {temp:.4f}, SIZE: {score.size}, PEN: {score.pen}")
            temp = self.cool(i)
        return state

    def score(self, nodes: set[Node], module_size: int) -> Score:
        """Calculates measure of size for node set and penalty within bootstraps."""
        size = len(nodes)
        n_mismatch = [
            min(len(nodes.difference(module)) for module in replicate)
            for replicate in self.bootstraps
        ]
        n_pen = int(len(self.bootstraps) * (1 - self.config["conf"]))
        pen = sum(sorted(n_mismatch)[:(n_pen - 1)]) * self.config["pen_weight"] * module_size
        return Score(size, pen)

    def do_accept_state(self, score: Score, new_score: Score, temp: float) -> bool:
        """Checks if a new state should be accepted."""
        delta_score = new_score.size - new_score.pen - (score.size - score.pen)
        if delta_score > 0:
            return True
        if np.exp(delta_score / temp) >= self.rng.uniform(0, 1):
            # Metropolis–Hastings algorithm
            return True
        return False

    def cool(self, i: int) -> float:
        """Applies exponential cooling schedule."""
        return self.config["temp_init"] * np.exp(-i * self.config["cool_rate"])

    def flip(self, nodes: set[Node], node: Node) -> set[Node]:
        """Flips membership of a node in a node set."""
        new_nodes = nodes.copy()
        if node in new_nodes:
            new_nodes.discard(node)
        else:
            new_nodes.add(node)
        return new_nodes

    def initialize_state(self, nodes: list[Node]) -> set[Node]:
        """Initializes candidate core."""
        num_init = self.rng.integers(1, len(nodes))
        self.rng.shuffle(nodes)
        return set(nodes[:(num_init - 1)])
