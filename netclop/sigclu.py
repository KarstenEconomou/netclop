"""Defines the SigClu class."""
from collections import namedtuple
from dataclasses import dataclass
from functools import cached_property
from os import PathLike
from typing import Optional

import numpy as np

from .constants import Node, Partition, SEED
from .exceptions import MissingResultError
from .upsetplot import UpSetPlot

type Size = int
Score = namedtuple("Score", ["size", "pen"])

class SigClu:
    """Finds robust cores of network partitions through recursive significance clustering."""
    @dataclass(frozen=True)
    class Config:
        seed: int = SEED
        sig: float = 0.05
        temp_init: float = 10.0
        cooling_rate: float = 0.9
        decay_rate: float = 1.0
        pen_scalar: float = 0.05
        min_core_size: Size = 6
        num_trials: int = 2
        num_exhaustion_loops: int = 20
        max_sweeps: int = 1000
        verbose: bool = True

    def __init__(self, partitions: list[Partition], **config_options):
        self.partitions = partitions
        self.cfg = self.Config(**config_options)

        self.rng = np.random.default_rng(self.cfg.seed)

        self.cores: Optional[Partition] = None

    def _log(self, msg: str):
        """Log message."""
        if self.cfg.verbose:
            print(msg)

    @cached_property
    def nodes(self) -> set[Node]:
        return set().union(*[node for partition in self.partitions for node in partition])

    @cached_property
    def n_pen(self) -> int:
        """Calculate the number of partitions to consider when penalizing."""
        return np.ceil(len(self.partitions) * (1 - self.cfg.sig)).astype(int)

    def run(self) -> None:
        """Find robust cores."""
        cores = []

        # Loop to find each core above min size threshold
        avail_nodes = self.nodes.copy()
        while True:
            if len(avail_nodes) < self.cfg.min_core_size:
                break

            core = self._find_core_sanitized(avail_nodes)
            if core is not None:
                avail_nodes.difference_update(core)  # Nodes in core are not available in future iters
                self._add_core(core, cores)
                self._sort_by_size(cores)
            else:
                break

        self.cores = cores

    def upset(self, path: PathLike, **kwargs) -> None:
        """Make an UpSet plot of cores."""
        if self.cores is None:
            raise MissingResultError()

        upset = UpSetPlot(self.cores, self.partitions, sig=self.cfg.sig, **kwargs)
        upset.plot(path)

    def _add_core(self, core: set[Node], cores: Partition) -> None:
        """Merge a core with a larger core if possible."""
        if len(cores) == 0:
            cores.append(core)
            return

        for i, prev_core in enumerate(cores):  # Cores is in descending order of size
            merged_core = core.union(prev_core)
            if self._all_form_core(merged_core):
                cores[i] = merged_core
                return
        cores.append(core)

    def _sort_by_size(self, nodes: Partition) -> None:
        """Manually sort cores from largest to smallest."""
        nodes.sort(key=self._measure_size, reverse=True)

    def _find_core_sanitized(self, nodes: set[Node], exhaustion_search: bool=True) -> Optional[set[Node]]:
        """Perform simulated annealing with wrapper for restarts."""
        if self._is_trivial(nodes) or self._all_form_core(nodes):
            return nodes

        best_state, best_score = {}, 0
        for _ in range(self.cfg.num_trials):
            state, (size, pen) = self._find_core(nodes)
            score = size - pen
            if score > best_score and pen == 0:
                best_state, best_score = state, score

        if self._measure_size(best_state) < self.cfg.min_core_size:
            # Best state is not of substantial size to be labelled a core
            # Begin exhaustion search to try to find small, but above threshold cores
            if exhaustion_search:
                for _ in range(self.cfg.num_exhaustion_loops):
                    best_state = self._find_core_sanitized(nodes, exhaustion_search=False)
                    if best_state is not None:
                        return best_state
            return None

        return best_state

    def _find_core(self, nodes: set[Node]) -> tuple[set[Node], Score]:
        """Find the largest core of node set through simulated annealing."""
        pen_weighting = self.cfg.pen_scalar * self._measure_size(nodes)
        nodes = list(nodes)

        # Initialize state
        state = self._initialize_state(nodes)
        score = self._score(state, pen_weighting)
        temp = self.cfg.temp_init

        # Core loop
        for t in range(self.cfg.max_sweeps):
            did_accept = False

            num_repetitions = 2 * self._num_repetitions(t, len(nodes))
            for _ in range(num_repetitions):
                # Generate trial state
                node = self.rng.choice(nodes)
                trial_state = self._flip(state, node)
                trial_score = self._score(trial_state, pen_weighting)

                # Query accepting trial state
                if self._do_accept_state(score, trial_score, temp):
                    state, score = trial_state, trial_score
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
        """Calculate a measure of size on a node set."""
        return len(nodes)

    def _score(self, nodes: set[Node], pen_weighting: float) -> Score:
        """Calculate measure of size for node set and penalty across partitions."""
        size = self._measure_size(nodes)

        n_mismatch = [
            min(self._measure_size(nodes.difference(module)) for module in replicate)
            for replicate in self.partitions
        ]
        # Only penalize the best n_pen partitions
        pen = sum(sorted(n_mismatch)[:self.n_pen]) * pen_weighting

        return Score(size, pen)

    def _do_accept_state(self, score: Score, trial_score: Score, temp: float) -> bool:
        """Check if a trial state should be accepted."""
        delta_score = (trial_score.size - trial_score.pen) - (score.size - score.pen)
        if delta_score > 0:
            return True
        elif np.exp(delta_score / temp) >= self.rng.uniform(0, 1):
            # Metropolis–Hastings algorithm
            return True
        else:
            return False

    def _cool(self, t: int) -> float:
        """Apply exponential cooling schedule."""
        return self.cfg.temp_init * (self.cfg.cooling_rate ** t)

    def _num_repetitions(self, t: int, n: int) -> int:
        """Apply exponential repetition schedule."""
        return np.ceil(n * (self.cfg.decay_rate ** t)).astype(int)

    def _initialize_state(self, nodes: list[Node]) -> set[Node]:
        """Initialize candidate core."""
        num_init = self.rng.integers(1, len(nodes))
        self.rng.shuffle(nodes)
        return set(nodes[:(num_init - 1)])

    def _all_form_core(self, nodes: set[Node]) -> bool:
        """Check if every node forms a core."""
        _, pen = self._score(nodes, 1)
        return pen == 0

    @staticmethod
    def _is_trivial(nodes: set[Node]) -> bool:
        """Check if a set of nodes are trivial."""
        return len(nodes) <= 1

    @staticmethod
    def _flip(nodes: set[Node], node: Node) -> set[Node]:
        """Flip membership of a node in a node set."""
        candidate = nodes.copy()
        if node in candidate:
            candidate.discard(node)
        else:
            candidate.add(node)
        return candidate
