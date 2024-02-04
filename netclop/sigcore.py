"""Defines classes for significance clustering."""
import dataclasses
from collections import namedtuple

import numpy as np

from .config_loader import load_config
from .constants import Node, Partition

Score = namedtuple("Score", ["size", "pen"])

@dataclasses.dataclass
class SigClu:
    """Finds significant core of nodes within a module."""
    partition: Partition
    bootstraps: tuple[Partition, ...]

    _config: dict[str, any] = dataclasses.field(default_factory=lambda: load_config()["sig_clu"])

    _rng: np.random.Generator = dataclasses.field(init=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self._config["seed"])

    def run(self) -> list[set[Node]]:
        """Finds the significant cores of all modules."""
        cores = []
        for i, module in enumerate(self.partition, 1):
            core = self.find_sig_core(module)
            cores.append(core)
            print(f"Sig core of module {i} found: {len(core)}/{len(module)} nodes")
        return cores

    def find_sig_core(self, module: set[Node]) -> set[Node]:
        """Finds significant core of a module."""
        num_nodes = len(module)
        if num_nodes <= 1:
            return module
        module = list(module)

        pen_weighting = self._config["pen_weight"] * num_nodes

        # Initialize state
        state = self._initialize_state(module)
        score = self._score(state, pen_weighting)
        temp = self._config["temp_init"]

        # Core loop
        for i in range(self._config["iter_max"]):
            did_accept = False
            for _ in range(num_nodes):
                # Flip one random node's membership from candidate state and score
                node = self._rng.choice(module)
                new_state = self._flip(state, node)
                new_score = self._score(new_state, pen_weighting)

                # Query accepting perturbed state
                if self._do_accept_state(score, new_score, temp):
                    state = new_state
                    score = new_score
                    did_accept = True

            if not did_accept:
                break
            temp = self._cool(i)
        return state

    def _score(self, nodes: set[Node], pen_weighting: float) -> Score:
        """Calculates measure of size for node set and penalty within bootstraps."""
        size = len(nodes)
        n_mismatch = [
            min(len(nodes.difference(module)) for module in replicate)
            for replicate in self.bootstraps
        ]
        n_pen = int(len(self.bootstraps) * (1 - self._config["sig"]))
        pen = sum(sorted(n_mismatch)[:(n_pen - 1)]) * pen_weighting
        return Score(size, pen)

    def _do_accept_state(self, score: Score, new_score: Score, temp: float) -> bool:
        """Checks if a new state should be accepted."""
        delta_score = new_score.size - new_score.pen - (score.size - score.pen)
        if delta_score > 0:
            return True
        if np.exp(delta_score / temp) >= self._rng.uniform(0, 1):
            # Metropolis–Hastings algorithm
            return True
        return False

    def _cool(self, i: int) -> float:
        """Applies exponential cooling schedule."""
        return self._config["temp_init"] * np.exp(-(i + 1) * self._config["cool_rate"])

    @staticmethod
    def _flip(nodes: set[Node], node: Node) -> set[Node]:
        """Flips membership of a node in a node set."""
        new_nodes = nodes.copy()
        if node in new_nodes:
            new_nodes.discard(node)
        else:
            new_nodes.add(node)
        return new_nodes

    def _initialize_state(self, nodes: list[Node]) -> set[Node]:
        """Initializes candidate core."""
        num_init = self._rng.integers(1, len(nodes))
        self._rng.shuffle(nodes)
        return set(nodes[:(num_init - 1)])
