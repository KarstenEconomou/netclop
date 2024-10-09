"""Defines classes for significance clustering."""
import dataclasses
import typing
from collections import namedtuple

import numpy as np

from .config_loader import load_config
from .constants import Node, Partition

Score = namedtuple("Score", ["size", "pen"])


@dataclasses.dataclass
class SigClu:
    """Finds robust cores of network partitions."""
    nodes: set[Node]
    partitions: typing.Sequence[Partition]

    _cfg: dict[str, any] = dataclasses.field(default_factory=lambda: load_config()["sig_clu"])

    _rng: np.random.Generator = dataclasses.field(init=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self._cfg["seed"])

    def run(self) -> list[set[Node]]:
        """Finds robust cores."""
        cores = []

        # Initialize
        avail_nodes = self.nodes.copy()

        # Loop to find each core above min size threshold
        while True:
            core = self._find_core_sanitized(avail_nodes)
            if len(core) >= self._cfg["thresh"]:
                cores.append(core)
                avail_nodes.difference_update(core)  # Remove nodes in core
            else:
                break

        self._sort_cores(cores)
        return cores

    def _sort_cores(self, cores: list[set[Node]]) -> None:
        """Manually sorts cores from largest to smallest."""
        cores.sort(key=self._measure_size, reverse=True)

    def _find_core_sanitized(self, nodes: set[Node]) -> set[Node]:
        """Performs simulated annealing with wrapper for restarts."""
        if self._is_trivial(nodes) or self._all_form_core(nodes):
            return nodes

        best_state = {}
        best_score = 0
        for _ in range(self._cfg["loops"]):
            state, (size, pen) = self._find_core(nodes)
            score = size - pen
            if score > best_score and pen == 0:
                best_state = state
                best_score = score
                if len(best_state) >= self._cfg["thresh"]:
                    break
        return best_state

    def _find_core(self, nodes: set[Node]) -> tuple[set[Node], Score]:
        """Simulated annealing to find the largest core of node set."""
        pen_weighting = self._cfg["pen_scalar"] * self._measure_size(nodes)
        nodes = list(nodes)

        # Initialize state
        state = self._initialize_state(nodes)
        score = self._score(state, pen_weighting)
        temp = self._cfg["temp_init"]

        # Core loop
        for t in range(self._cfg["max_sweeps"]):
            did_accept = False

            num_repetitions = 2 * self._num_repetitions(t, len(nodes))
            for _ in range(num_repetitions):
                # Generate trial state
                node = self._rng.choice(nodes)
                trial_state = self._flip(state, node)
                trial_score = self._score(trial_state, pen_weighting)

                # Query accepting trial state
                if self._do_accept_state(score, trial_score, temp):
                    state = trial_state
                    score = trial_score
                    did_accept = True

            if not did_accept:
                break
            temp = self._cool(t)

        # One riffle through unassigned nodes
        for node in set(nodes).difference(state):
            trial_state = self._flip(state, node)
            trial_score = self._score(trial_state, pen_weighting)
            if trial_score.pen == 0:
                state = trial_state

        return state, score

    def _measure_size(self, nodes: set[Node]) -> int | float:
        """Calculates a measure of size on a node set."""
        return len(nodes)

    def _score(self, nodes: set[Node], pen_weighting: float) -> Score:
        """Calculates measure of size for node set and penalty within bootstraps."""
        size = self._measure_size(nodes)

        n_mismatch = [
            min(self._measure_size(nodes.difference(module)) for module in replicate)
            for replicate in self.partitions
        ]
        n_pen = int(len(self.partitions) * (1 - self._cfg["sig"]))
        pen = sum(sorted(n_mismatch)[:(n_pen - 1)]) * pen_weighting

        return Score(size, pen)

    def _do_accept_state(self, score: Score, trial_score: Score, temp: float) -> bool:
        """Checks if a trial state should be accepted."""
        delta_score = (trial_score.size - trial_score.pen) - (score.size - score.pen)
        if delta_score > 0:
            return True
        elif np.exp(delta_score / temp) >= self._rng.uniform(0, 1):
            # Metropolis–Hastings algorithm
            return True
        else:
            return False

    def _cool(self, t: int) -> float:
        """Applies exponential cooling schedule."""
        return self._cfg["temp_init"] * (self._cfg["cool_rate"] ** t)

    def _num_repetitions(self, t: int, n: int) -> int:
        """Applies exponential repetition schedule."""
        return np.ceil(n * (self._cfg["decay_rate"] ** t)).astype(int)

    def _initialize_state(self, nodes: list[Node]) -> set[Node]:
        """Initializes candidate core."""
        num_init = self._rng.integers(1, len(nodes))
        self._rng.shuffle(nodes)
        return set(nodes[:(num_init - 1)])

    def _all_form_core(self, nodes: set[Node]) -> bool:
        """Checks if every node forms a core."""
        (_, pen) = self._score(nodes, self._cfg["pen_scalar"] * len(nodes))
        return pen == 0

    @staticmethod
    def _is_trivial(nodes: set[Node]) -> bool:
        """Checks if a set of nodes are trivial."""
        return len(nodes) <= 1

    @staticmethod
    def _flip(nodes: set[Node], node: Node) -> set[Node]:
        """Flips membership of a node in a node set."""
        candidate = nodes.copy()
        if node in candidate:
            candidate.discard(node)
        else:
            candidate.add(node)
        return candidate
