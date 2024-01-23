""""""
import dataclasses
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from infomap import Infomap
from networkx import DiGraph

type Node = str
type Partition = tuple[set[Node], ...]

@dataclasses.dataclass
class Bootstrap:
    """Class for partitioning and bootstrapping network."""
    network: DiGraph

    seed: int = 42
    markov_time: float = 2
    im_trials: int = 10

    n_replicates: int = 1000
    conc_scale: float = 30

    rng: np.random.Generator = np.random.default_rng(seed)

    def run(self) -> tuple[Partition, tuple[Partition, ...]]:
        """Run parametric bootstrap."""
        network = self.network
        partition, _ = self.partition_network(network)

        replicate_networks = self.resample_weights(network)
        replicate_partitions = tuple(self.partition_network(rep)[0] for rep in replicate_networks)

        return partition, replicate_partitions

    def partition_network(self, network: DiGraph) -> tuple[Partition, pd.DataFrame]:
        """Partition one network with Infomap."""
        im = Infomap(
            silent=True,
            two_level=True,
            flow_model="directed",
            num_trials=self.im_trials,
            markov_time=self.markov_time,
            seed=self.seed,
        )
        _ = im.add_networkx_graph(network)
        im.run()

        node_info = im.get_dataframe(["name", "module_id", "flow", "modular_centrality"])
        partition = self.get_module_node_map(node_info)

        return partition, node_info
    
    def resample_weights(self, network: DiGraph) -> tuple[DiGraph, ...]:
        """Resample out-edge weight distributions to populate replicate networks."""
        network = self.network

        # Duplicate network as replicates
        replicate_nets = [network.copy() for _ in range(self.n_replicates)]

        for u in network.nodes:
            # Resample out-edge weight distribution of node for all replicate networks
            alpha = tuple(self.conc_scale * w for _, _, w in network.out_edges(u, data="weight"))
            samples = self.rng.dirichlet(alpha, size=self.n_replicates)

            # Assign edge weights
            for replicate_net, sample in zip(replicate_nets, samples):
                for i, (_, v) in enumerate(network.out_edges(u)):
                    replicate_net[u][v]["weight"] = sample[i]

        return replicate_nets
    
    def get_module_node_map(self, node_info: pd.DataFrame) -> Partition:
        """Create set of sets partition representation from node-module index pairs."""
        return tuple(node_info.groupby("module_id")["name"].apply(set))

    @classmethod
    def from_file(cls, in_path: Path) -> Self:
        """Read a file of tail, head, weight to generate network representation."""
        df = pd.read_csv(
            in_path,
            names=["tail", "head", "weight"],
            index_col=False,
            skiprows=1,
            delim_whitespace=True,
        )

        network = DiGraph()
        network.add_weighted_edges_from((u, v, w) for u, v, w in df.itertuples(index=False))
        return cls(network)
