"""NetworkEnsemble class."""
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from os import PathLike
from typing import Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from infomap import Infomap

from netclop.centrality import centrality_registry
from netclop.constants import SEED, WEIGHT_ATTR
from netclop.ensemble.netutils import flatten_partition, label_partition
from netclop.ensemble.sigclu import SigClu
from netclop.exceptions import MissingResultError
from netclop.log import Logger
from netclop.typing import NodeMetric, NodeSet, Partition


class NetworkEnsemble:
    """Network operations involving an ensemble of networks."""
    @dataclass(frozen=True)
    class Config:
        seed: int = SEED
        num_bootstraps: int = 1000
        im_markov_time: float = 1.0
        im_variable_markov_time: bool = True
        im_num_trials: int = 5

    def __init__(
        self,
        net: nx.DiGraph | Sequence[nx.DiGraph],
        logger: Logger = None,
        silent: bool = False,
        **config_options
    ):
        self.logger = Logger(silent=silent) if logger is None else logger
        self.cfg = self.Config(**config_options)

        self.nets = net if isinstance(net, Sequence) else [net]

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

    def to_nodelist(self, metrics: Optional[dict[str, NodeMetric]] = None, path: PathLike = None) -> pd.DataFrame:
        """Create a node list."""
        df = pd.DataFrame({"node": list(self.nodes)})

        if self.cores is not None:
            df["core"] = df["node"].map(label_partition(self.cores)).fillna(0).astype(int)

        if metrics is not None:
            for index, value in metrics.items():
                df[index] = df["node"].map(value)

        if path is not None:
            df.to_csv(path, index=False)

        return df

    def is_ensemble(self) -> bool:
        """Check if an ensemble of nets is stored."""
        return len(self.nets) > 1

    def is_bootstrapped(self) -> bool:
        """Check if replicate networks have been bootstrapped."""
        return self.bootstraps is not None

    def partition(self) -> None:
        """Partition networks."""
        if self.is_ensemble():
            nets = self.nets
        else:
            if not self.is_bootstrapped():
                self.bootstrap(self.nets[0])
            nets = self.bootstraps

        self.logger.log(
            f"Partitioning {len(nets)} networks with Infomap: "
            f"mt {self.cfg.im_markov_time} {"(variable)" if self.cfg.im_variable_markov_time else "(static)"}"
        )
        self.partitions = [
            self.im_partition(net) for net in self.logger.pbar(nets, desc="Community detection", unit="net")
        ]
        self.logger.log(f"{self.logger.stat([len(part) for part in self.partitions])} modules")

    def im_partition(self, net: nx.DiGraph) -> Partition:
        """Partition a network with Infomap."""
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
        self.logger.log(f"Resampling {self.cfg.num_bootstraps} networks.")
        edges, weights = zip(*nx.get_edge_attributes(net, WEIGHT_ATTR).items())
        weights = np.array(weights)
        num_edges = len(edges)

        rng = np.random.default_rng(self.cfg.seed)
        new_weights = rng.poisson(lam=weights.reshape(1, -1), size=(self.cfg.num_bootstraps, num_edges))

        bootstraps = []
        for i in self.logger.pbar(range(self.cfg.num_bootstraps), desc="Net construction", unit="net"):
            bootstrap = net.copy()
            edge_attrs = {edges[j]: {WEIGHT_ATTR: new_weights[i, j]} for j in range(num_edges)}
            nx.set_edge_attributes(bootstrap, edge_attrs)
            bootstraps.append(bootstrap)
        self.bootstraps = bootstraps

    def sigclu(self, upset_config: dict = None, **kwargs) -> None:
        """Computes recursive significance clustering on partition ensemble."""
        if self.partitions is None:
            self.partition()

        sc = SigClu(
            self.partitions,
            logger=self.logger,
            **kwargs
        )
        sc.run()
        self.cores = sc.cores

        if upset_config is not None:
            self.logger.log("Calculating coalescence frequency and generating UpSet plot.")
            sc.upset(**upset_config)

    def node_centrality(self, name: str, use_bootstraps: bool = False, **kwargs) -> NodeMetric:
        """Compute node centrality indices."""
        index = centrality_registry.get(name)

        if use_bootstraps and not self.is_bootstrapped():
            raise MissingResultError()

        if self.is_ensemble() or use_bootstraps:
            centrality_list: list[NodeMetric] = []

            nets = self.nets if not use_bootstraps else self.bootstraps
            for net in nets:
                centrality_list.append(index.compute(net, **kwargs))

            return self.avg_node_centrality(centrality_list)
        else:
            return index.compute(self.nets[0], **kwargs)

    @staticmethod
    def avg_node_centrality(node_centralities: list[NodeMetric]) -> NodeMetric:
        """Average the centrality index of each node."""
        centrality_sums = defaultdict(float)
        node_counts = defaultdict(int)

        for centrality_dict in node_centralities:
            for node, value in centrality_dict.items():
                centrality_sums[node] += value
                node_counts[node] += 1

        return dict((node, centrality_sums[node] / node_counts[node]) for node in centrality_sums)
