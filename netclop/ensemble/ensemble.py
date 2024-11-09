"""NetworkEnsemble class."""
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Sequence

import numpy as np
from infomap import Infomap

from netclop.constants import SEED
from netclop.ensemble.netutils import flatten_partition
from netclop.ensemble.centrality import *
from netclop.ensemble.sigclu import SigClu
from netclop.exceptions import MissingResultError
from netclop.typing import NodeSet, Partition


class NetworkEnsemble:
    """Network operations involving an ensemble of networks."""
    @dataclass(frozen=True)
    class Config:
        seed: int = SEED
        num_bootstraps: int = 1000
        im_markov_time: float = 1.0
        im_variable_markov_time: bool = True
        im_num_trials: int = 5

    def __init__(self, net: nx.DiGraph | Sequence[nx.DiGraph], **config_options):
        self.nets = net if isinstance(net, Sequence) else [net]
        self.cfg = self.Config(**config_options)

        self.bootstraps: Optional[list[nx.DiGraph]] = None
        self.partitions: Optional[list[Partition]] = None
        self.cores: Optional[Partition] = None

    @cached_property
    def nodes(self) -> NodeSet:
        return frozenset().union(*[net.nodes for net in self.nets])

    @property
    def unstable_nodes(self) -> NodeSet:
        if self.cores is None:
            raise MissingResultError()
        return self.nodes.difference(flatten_partition(self.cores))

    def is_ensemble(self) -> bool:
        """Check if an ensemble of nets is stored."""
        return len(self.nets) > 1

    def is_bootstrapped(self) -> bool:
        """Check if replicate networks have been bootstrapped."""
        return len(self.bootstraps) == self.cfg.num_bootstraps

    def partition(self) -> None:
        """Partition networks."""
        if self.is_ensemble():
            self.partitions = [self.im_partition(net) for net in self.nets]
        else:
            self.bootstrap(self.nets[0])
            self.partitions = [self.im_partition(bootstrap) for bootstrap in self.bootstraps]

    def im_partition(self, net: nx.DiGraph) -> Partition:
        """Partitions a network."""
        im = Infomap(
            silent=True,
            two_level=True,
            flow_model="directed",
            seed=self.cfg.seed,
            num_trials=self.cfg.im_num_trials,
            markov_time=self.cfg.im_markov_time,
            variable_markov_time=self.cfg.im_variable_markov_time,
        )
        _ = im.add_networkx_graph(net, weight="weight")
        im.run()

        partition = im.get_dataframe(["name", "module_id"]).groupby("module_id")["name"].apply(set).tolist()
        return partition

    def bootstrap(self, net: nx.DiGraph) -> None:
        """Resample edge weights."""
        edges, weights = zip(*nx.get_edge_attributes(net, 'weight').items())
        weights = np.array(weights)
        num_edges = len(edges)

        rng = np.random.default_rng(self.cfg.seed)
        new_weights = rng.poisson(lam=weights.reshape(1, -1), size=(self.cfg.num_bootstraps, num_edges))

        bootstraps = []
        for i in range(self.cfg.num_bootstraps):
            bootstrap = net.copy()
            edge_attrs = {edges[j]: {"weight": new_weights[i, j]} for j in range(num_edges)}
            nx.set_edge_attributes(bootstrap, edge_attrs)
            bootstraps.append(bootstrap)
        self.bootstraps = bootstraps

    def sigclu(self, upset_config: dict=None, **kwargs) -> None:
        """Computes recursive significance clustering on partition ensemble."""
        if self.partitions is None:
            self.partition()

        sc = SigClu(
            self.partitions,
            **kwargs
        )
        sc.run()
        self.cores = sc.cores

        if upset_config is not None:
            sc.upset(**upset_config)

    def node_centrality(self, centrality_index: str, use_bootstraps: bool=False, **kwargs) -> CentralityNodes:
        """Compute node centrality indices."""
        centrality_functions = {
            "out-degree": nx.out_degree_centrality,
            "in-degree": nx.in_degree_centrality,
            "out-strength": out_strength,
            "in-strength": in_strength,
            "betweenness": nx.betweenness_centrality,
            "pagerank": nx.pagerank,
        }
        if not (centrality_func := centrality_functions.get(centrality_index.lower())):
            raise ValueError(f"Unknown centrality index: {centrality_index}")

        if use_bootstraps and not self.is_bootstrapped():
            raise MissingResultError()

        if self.is_ensemble() or use_bootstraps:
            centrality_list: list[CentralityNodes] = []

            nets = self.nets if not use_bootstraps else self.bootstraps
            for net in nets:
                centrality_list.append(centrality_func(net, **kwargs))

            return self.avg_node_centrality(centrality_list)
        else:
            return centrality_func(self.nets[0], **kwargs)

    @staticmethod
    def avg_node_centrality(node_centralities: list[CentralityNodes]) -> CentralityNodes:
        """Average the centrality index of each node."""
        centrality_sums = defaultdict(float)
        node_counts = defaultdict(int)

        # Sum centrality values and count occurrences for each node
        for centrality_dict in node_centralities:
            for node, value in centrality_dict.items():
                centrality_sums[node] += value
                node_counts[node] += 1

        # Compute average for each node
        avg_centrality = dict((node, centrality_sums[node] / node_counts[node]) for node in centrality_sums)

        return avg_centrality
