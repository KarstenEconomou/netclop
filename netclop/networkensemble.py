"""Defines the NetworkEnsemble class."""
from dataclasses import dataclass
from functools import cached_property
from os import PathLike
from typing import Optional

import networkx as nx
import numpy as np
from infomap import Infomap

from .constants import Partition, SEED
from .exceptions import MissingResultError
from .sigclu import SigClu


class NetworkEnsemble:
    """Network operations for creating an ensemble of partitions."""
    @dataclass(frozen=True)
    class Config:
        seed: int = SEED
        num_bootstraps: int = 1000
        im_markov_time: float = 1.0
        im_variable_markov_time: bool = True
        im_num_trials: int = 5

    def __init__(self, net: nx.DiGraph | list[nx.DiGraph], **config_options):
        self.net = net if isinstance(net, list) else [net]
        self.cfg = self.Config(**config_options)

        self.bootstraps: Optional[list[nx.DiGraph]] = None
        self.partitions: Optional[list[Partition]] = None
        self.cores: Optional[Partition] = None

    @cached_property
    def nodes(self):
        return set().union(*[net.nodes for net in self.net])

    def partition(self):
        if len(self.net) == 1:
            self.make_bootstrap_ensemble(self.net[0])
            self.partitions = [self.im_partition(bootstrap) for bootstrap in self.bootstraps]
        else:
            self.partitions = [self.im_partition(net) for net in self.net]

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

    def make_bootstrap_ensemble(self, net: nx.DiGraph) -> None:
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
            raise MissingResultError()

        sc = SigClu(
            self.partitions,
            **kwargs
        )
        sc.run()
        self.cores = sc.cores

        if upset_config is not None:
            sc.upset(**upset_config)

    def _is_bootstrapped(self) -> bool:
        return len(self.bootstraps) == self.cfg.num_bootstraps