"""Defines the NetworkOps class."""
import csv
import dataclasses
import typing

import h3.api.numpy_int as h3
import networkx as nx
import numpy as np
import pandas as pd
from infomap import Infomap

from .config_loader import load_config
from .constants import Node, Partition, Path
from .sigcore import SigClu

CONFIG = load_config()

@dataclasses.dataclass
class NetworkOps:
    """Network operations."""
    _config: dict[str, any] = dataclasses.field(default_factory=lambda: load_config())

    def to_dataframe(self, net: nx.DiGraph, out_path: Path = None) -> pd.DataFrame:
        """Writes the network nodelist with attributes."""
        df = pd.DataFrame.from_dict(dict(net.nodes(data=True)), orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={"index": "node"}, inplace=True)
        if out_path is not None:
            df.to_csv(out_path, index=True)
        return df

    def write_edgelist(self, net: nx.DiGraph, path: Path) -> None:
        """Writes the network edgelist to file."""
        nx.write_edgelist(
            net,
            path,
            delimiter=",",
            comments="#",
            data=["weight", "weight_norm"],
        )

    def from_file(self, path: Path) -> nx.DiGraph:
        """Constructs NetworkOps from edge list file."""
        net = nx.read_edgelist(
            path,
            comments="#",
            delimiter=",",
            create_using=nx.DiGraph,
            nodetype=str,
            data=[("weight", float), ("weight_norm", float)],
        )
        return net

    def from_positions(self, path: Path) ->  nx.DiGraph:
        """Constructs a network from file of initial and final coordinates."""
        data = pd.read_csv(
            path,
            names=["initial_lng", "initial_lat", "final_lng", "final_lat"],
            index_col=False,
            comment="#",
        )

        resolution = self._config["binning"]["res"]

        def bin_positions(
            lngs: typing.Sequence[float],
            lats: typing.Sequence[float],
            res: int,
        ) -> list[int]:
            """Bins (lng, lat) coordinate pairs into an H3 cell."""
            return [h3.latlng_to_cell(lat, lng, res) for lat, lng in zip(lats, lngs)]

        print(f"Binning {data.shape[0]} particle positions")
        srcs = bin_positions(data["initial_lng"], data["initial_lat"], resolution)
        tgts = bin_positions(data["final_lng"], data["final_lat"], resolution)
        edges = tuple(zip(srcs, tgts))
        return self.from_edgelist(edges)

    def from_edgelist(self, edges: typing.Sequence[tuple[str, str]]) -> typing.Self:
        """Constructs NetworkOps from edge list."""
        net = nx.DiGraph()
        for src, tgt in edges:
            if net.has_edge(src, tgt):
                # Record another transition along a recorded edge
                net[src][tgt]["weight"] += 1
            else:
                # Record a new edge
                net.add_edge(src, tgt, weight=1)

        self.normalize_edge_weights(net)

        nx.relabel_nodes(net, dict((name, str(name)) for name in net.nodes), copy=False)
        print(f"Constructed network of {len(net.nodes)} nodes and {len(net.edges)} edges")
        return net

    def compute_node_measures(self, net: nx.DiGraph, cores: list[set[Node]]=None) -> None:
        """Calculate node measures and save as attributes."""
        # Significance
        significant_nodes = set.union(*cores)
        for node in net.nodes():
            net.nodes[node]["significant"] = node in significant_nodes

        # In- and out- degree and strength
        in_degs = dict(net.in_degree())
        out_degs = dict(net.out_degree())

        in_strs = dict(net.in_degree(weight='weight'))
        out_strs = dict(net.out_degree(weight='weight'))

        # Betweenness
        betweenness_centrality = nx.betweenness_centrality(net)

        # Update network with metrics
        nx.set_node_attributes(net, in_degs, "in_deg")
        nx.set_node_attributes(net, out_degs, "out_deg")
        nx.set_node_attributes(net, in_strs, "in_str")
        nx.set_node_attributes(net, out_strs, "out_str")
        nx.set_node_attributes(net, betweenness_centrality, "betweenness")

    def normalize_edge_weights(self, net: nx.DiGraph) -> None:
        """Normalizes out-edge weight distributions to sum to unity."""
        for u in net.nodes:
            out_wgt = sum(weight for _, _, weight in net.out_edges(u, data='weight', default=0))
            for v in net.successors(u):
                net[u][v]["weight_norm"] = net[u][v]["weight"] / out_wgt if out_wgt != 0 else 0

    def partition(self, net: nx.DiGraph, node_info: bool = True) -> None:
        """Partitions a network."""
        im = Infomap(
            silent=True,
            two_level=True,
            flow_model="directed",
            seed=self._config["infomap"]["seed"],
            num_trials=self._config["infomap"]["num_trials"],
            markov_time=self._config["infomap"]["markov_time"],
            variable_markov_time=self._config["infomap"]["variable_markov_time"],
        )
        _ = im.add_networkx_graph(net, weight="weight")
        im.run()

        if node_info:
            node_info = im.get_dataframe(["name", "module_id", "flow", "modular_centrality"])
            print(f"Partitioned into {len(node_info["module_id"].unique())} modules")
        else:
            node_info = im.get_dataframe(["name", "module_id"])
        node_info = node_info.rename(columns={"name": "node", "module_id": "module"})

        modular_desc = node_info.set_index("node").to_dict(orient="index")
        nx.set_node_attributes(net, modular_desc)

    def make_bootstraps(self, net: nx.DiGraph) -> tuple[nx.DiGraph, ...]:
        """Resample edge weights."""
        edges, weights = zip(*nx.get_edge_attributes(net, 'weight').items())
        weights = np.array(weights)
        num_edges = len(edges)

        num_bootstraps = self._config["bootstrap"]["num_bootstraps"]

        rng = np.random.default_rng(self._config["bootstrap"]["seed"])
        new_weights = rng.poisson(lam=weights.reshape(1, -1), size=(num_bootstraps, num_edges))

        bootstraps = []
        for i in range(num_bootstraps):
            bootstrap = net.copy()
            edge_attrs = {edges[j]: {"weight": new_weights[i, j]} for j in range(num_edges)}
            nx.set_edge_attributes(bootstrap, edge_attrs)
            bootstraps.append(bootstrap)

        print(f"Poisson-resampled {num_bootstraps} bootstrap networks")
        return bootstraps

    def significance_cluster(
        self,
        partition: Partition,
        bootstraps: tuple[Partition],
    ) -> list[set[Node]]:
        """Finds significant core of each module in the partition."""
        sig_clu = SigClu(partition, bootstraps, self._config["sig_clu"])
        return sig_clu.run()

    def group_nodes_by_module(self, net: nx.DiGraph) -> Partition:
        """Groups nodes in a network into sets by their module."""
        node_modules = list(net.nodes(data='module'))
        df = pd.DataFrame(node_modules, columns=["node", "module"])
        return df.groupby("module")["node"].apply(set).tolist()
