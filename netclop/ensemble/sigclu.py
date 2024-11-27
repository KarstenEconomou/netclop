"""SigClu class."""
from collections import namedtuple
from dataclasses import dataclass
from functools import cached_property
from os import PathLike
from typing import Optional

import numpy as np

from netclop.constants import SEED
from netclop.ensemble.netutils import flatten_partition
from netclop.ensemble.upsetplot import UpSetPlot
from netclop.exceptions import MissingResultError
from netclop.typing import Node, NodeSet, Partition
from netclop.log import Logger

type Size = int
Score = namedtuple("Score", ["size", "pen"], defaults=[0, 0])


class SigClu:
    """Finds robust cores of network partitions through significance clustering."""
    @dataclass(frozen=True)
    class Config:
        seed: int = SEED
        sig: float = 0.05
        temp_init: float = 1.0
        cooling_rate: float = 0.99
        decay_rate: float = 1.0
        pen_scalar: float = 2.0
        rep_scalar: int = 1
        min_core_size: Size = 6
        num_trials: int = 1
        num_exhaustion_loops: int = 10
        max_sweeps: int = 1000
        initialize_all: bool = True,

    def __init__(self, partitions: list[Partition], logger: Logger = None, silent: bool = False, **config_options):
        self.logger = Logger(silent=silent) if logger is None else logger
        self.cfg = self.Config(**config_options)

        self.partitions = partitions

        self.rng = np.random.default_rng(self.cfg.seed)

        self.cores: Optional[Partition] = None

    @cached_property
    def nodes(self) -> NodeSet:
        """Set of all nodes present in partitions."""
        return flatten_partition(self.partitions)

    @cached_property
    def n_pen(self) -> int:
        """Number of partitions to consider when penalizing."""
        return np.ceil(len(self.partitions) * (1 - self.cfg.sig)).astype(int)

    @cached_property
    def __node_ref_index(self) -> dict[Node, int]:
        """Mapping of node name to reference index based on underlying node names."""
        nodes_sorted = sorted(self.nodes, key=int)
        return dict((node, index) for index, node in enumerate(nodes_sorted))

    def run(self) -> None:
        """Find robust cores."""
        self.logger.log(
            f"Running recursive significance clustering on {len(self.partitions)} partitions: "
            f"level {self.cfg.sig}, init temp {self.cfg.temp_init}, cool rate {self.cfg.cooling_rate}, " 
            f"min size {self.cfg.min_core_size}"
        )
        cores = []

        # Loop to find each core above min size threshold
        avail_nodes = set(self.nodes)
        pbar = self.logger.make_pbar(desc="Significance clustering", unit="core")
        while len(avail_nodes) >= self.cfg.min_core_size:
            self.logger.pbar_info(pbar, f"{len(avail_nodes)}avail")
            core = self._find_core_sanitized(avail_nodes)
            if core is not None:
                avail_nodes.difference_update(core)  # Nodes in core are not available in future iters
                cores.append(core)
                self._sort_by_size(cores)

                self.logger.update_pbar(pbar)
            else:
                break

        self.logger.close_pbar(pbar)
        self.cores = cores
        self.logger.log(f"{len(cores)} cores, size: {', '.join(map(str, [len(core) for core in cores]))}")

    def _find_core_sanitized(self, nodes: NodeSet, exhaustion_search: bool=True) -> Optional[NodeSet]:
        """Perform simulated annealing with wrapper for restarts."""
        if self._is_trivial(nodes) or self._all_form_core(nodes):
            return nodes

        best_state, best_score = {}, 0
        for i in range(self.cfg.num_trials):
            state, (size, pen) = self._find_core(nodes)
            score = size - pen

            if score > best_score and pen == 0:
                best_state, best_score = state, score

        if best_score < self.cfg.min_core_size:
            # Best state is not of substantial size to be labelled a core
            # Begin exhaustion search to try to find small, but above threshold cores
            if exhaustion_search:
                for _ in self.logger.pbar(
                        range(self.cfg.num_exhaustion_loops),
                        desc="Exhaustion search",
                        leave=False,
                ):
                    best_state = self._find_core_sanitized(nodes, exhaustion_search=False)
                    if best_state is not None:
                        return best_state
            return None
        return best_state

    def _find_core(self, nodes: NodeSet) -> tuple[NodeSet, Score]:
        """Find the largest core of node set through simulated annealing."""
        pen_weighting = self._make_penalty_weight(nodes)
        nodes = self._nodeset_to_list_ordered(nodes)

        # Initialize state
        state = self._initialize_state(nodes)
        score = self._score(state, pen_weighting)
        temp = self.cfg.temp_init

        # Core loop
        for t in (pbar := self.logger.pbar(
            range(self.cfg.max_sweeps),
            length=False,
            leave=False,
        )):
            self.logger.pbar_info(pbar, f"{temp:.2f}temp, {score.size}size, {score.pen:.2f}pen")
            did_accept = False

            num_repetitions = self._num_repetitions(t, len(nodes))
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
        unassigned_nodes = self._nodeset_to_list_ordered(set(nodes).difference(state))
        self.rng.shuffle(unassigned_nodes)
        for node in unassigned_nodes:
            trial_state = self._flip(state, node)
            trial_score = self._score(trial_state, pen_weighting)
            if trial_score.pen == 0:
                state, score = trial_state, trial_score

        self.logger.pbar_info(pbar, f"{temp:.2f}temp, {score.size}size, {score.pen:.2f}pen")
        self.logger.close_pbar(pbar)

        return state, score

    def _measure_size(self, nodes: NodeSet) -> Size:
        """Calculate a measure of size on a node set."""
        return len(nodes)

    def _sort_by_size(self, cores: Partition) -> None:
        """Manually sort cores from largest to smallest."""
        cores.sort(key=self._measure_size, reverse=True)

    def _make_penalty_weight(self, nodes: NodeSet) -> float:
        """Calculate the weight of a penalty in scoring."""
        # Penalty difference should be on order of size difference of successive states
        pen_weight = self._measure_size(nodes) / self.n_pen

        # Scale by desired weight of penalty
        return self.cfg.pen_scalar * pen_weight

    def _score(self, nodes: NodeSet, pen_weighting: float) -> Score:
        """Calculate measure of size for node set and penalty across partitions."""
        size = self._measure_size(nodes)

        mismatch = [
            min(self._measure_size(nodes.difference(module)) for module in replicate)
            for replicate in self.partitions
        ]
        # Only penalize the best n_pen partitions
        pen = sum(sorted(mismatch)[:self.n_pen]) * pen_weighting

        return Score(size, pen)

    def _do_accept_state(self, score: Score, trial_score: Score, temp: float) -> bool:
        """Check if a trial state should be accepted."""
        delta_score = (trial_score.size - trial_score.pen) - (score.size - score.pen)
        if delta_score >= 0:  # Accept state if better or equal
            return True
        elif np.exp(delta_score / temp) >= self.rng.uniform(0, 1):
            # Metropolis–Hastings algorithm
            return True
        else:
            return False

    def _cool(self, t: int) -> float:
        """Apply exponential cooling schedule."""
        # return self.cfg.temp_init * (self.cfg.cooling_rate ** (t + 1))
        return self.cfg.temp_init * (self.cfg.cooling_rate ** (t + 1))

    def _num_repetitions(self, t: int, n: int) -> int:
        """Apply exponential repetition schedule."""
        return self.cfg.rep_scalar * n

    def _initialize_state(self, nodes: list[Node]) -> NodeSet:
        """
        Initialize candidate core.

        Generates the number of nodes to include in initial state and sample them.
        """
        if self.cfg.initialize_all:
            return set(nodes)

        num_init = self.rng.integers(1, len(nodes))
        self.rng.shuffle(nodes)
        return set(nodes[:(num_init - 1)])

    def _all_form_core(self, nodes: NodeSet) -> bool:
        """Check if every node forms a core."""
        _, pen = self._score(nodes, 1)
        return pen == 0

    def _nodeset_to_list_ordered(self, nodes: NodeSet) -> list[Node]:
        """Complete type conversion from set to a list ordered by reference index."""
        return sorted(nodes, key=lambda node: self.__node_ref_index[node])

    def upset(self, path: PathLike, **kwargs) -> None:
        """Make an UpSet plot of cores."""
        if self.cores is None:
            raise MissingResultError()

        upset = UpSetPlot(self.cores, self.partitions, sig=self.cfg.sig, **kwargs)
        upset.plot(path)

    @staticmethod
    def _is_trivial(nodes: NodeSet) -> bool:
        """Check if a set of nodes are trivial."""
        return len(nodes) <= 1

    @staticmethod
    def _flip(nodes: NodeSet, node: Node) -> NodeSet:
        """Flip membership of a node in a node set."""
        candidate = nodes.copy()
        if node in candidate:
            candidate.discard(node)
        else:
            candidate.add(node)
        return candidate
