"""Defines the NetworkOps class."""
import dataclasses
import typing

import h3.api.numpy_int as h3
import networkx as nx
import numpy as np
import pandas as pd
from infomap import Infomap

from .config_loader import load_config, update_config
from .constants import Node, Partition, Path
from .sigcore import SigClu, SigCluScheme

CONFIG = load_config()

@dataclasses.dataclass
class NetworkOps:
    """Network operations."""
    _config: dict[str, any] = dataclasses.field(default_factory=lambda: load_config())

    def update_config(self, cfg_update: dict) -> None:
        """Updates operation configuration."""
        update_config(self._config, cfg_update)

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
        return net

    def get_attr_list(self, net: nx.DiGraph, attr: str="module") -> set:
        """Gets set of attr."""
        return {net.nodes[node][attr] for node in net.nodes}

    def modular_strength(self, net: nx.DiGraph, attr: str="module") -> dict:
        """Calculates strength measures of each label."""
        labels = self.get_attr_list(net, attr)
        strength = {
            l: {"ext": {"out": 0, "in": 0}, "int": 0} for l in labels
        }

        for src, tgt, wgt in net.edges.data("weight"):
            # Out-edge of src, in-edge of tgt, with weight wgt
            src_l = net.nodes[src][attr]
            tgt_l = net.nodes[tgt][attr]
            if src_l != tgt_l:
                strength[src_l]["ext"]["out"] += wgt
                strength[tgt_l]["ext"]["in"] += wgt
            else:
                strength[src_l]["int"] += wgt

        return strength

    def coherence_fortress(self, strength: dict) -> tuple[dict[int, float], dict[int, float]]:
        """Computes the coherence and fortress of each label."""
        coherence = {}
        fortress = {}
        for l in strength.keys():
            out_str = strength[l]["int"] + strength[l]["ext"]["out"]
            in_str = strength[l]["int"] + strength[l]["ext"]["in"]

            coherence[l] = strength[l]["int"] / out_str if out_str != 0 else 0
            fortress[l] = strength[l]["int"] / in_str if in_str != 0 else 0
        return coherence, fortress

    def cohesion_mixing(self, net: nx.DiGraph, attr: str="module") -> tuple[float, float]:
        """Calculates cohesion and mixing of a partition."""
        strength = self.modular_strength(net, attr)
        labels = set(strength.keys()).difference({0})

        net_str = sum(strength[l]["int"] + strength[l]["ext"]["out"] for l in labels)
        net_int_str = sum(strength[l]["int"] for l in labels)

        cohesion = net_int_str / net_str

        label_mix = self.mixing(net, attr)
        mixing = sum(label_mix[l] * strength[l]["int"] for l in labels) / net_int_str

        return cohesion, mixing

    def mixing(self, net: nx.DiGraph, attr: str="module") -> dict[int, float]:
        """Calculates the mixing parameter of each label."""
        mixing = {}
        labels = self.group_nodes_by_attr(net, attr)
        for l in labels:
            retain_wgts = []
            for src in l:
                src_l = net.nodes[src][attr]
                for tgt in net.successors(src):
                    tgt_l = net.nodes[tgt][attr]
                    if src_l == tgt_l:
                        wgt = net.get_edge_data(src, tgt)["weight"]
                        retain_wgts.append(wgt)

            total_retain_wgt = sum(retain_wgts)
            norm_wgts = [wgt / total_retain_wgt for wgt in retain_wgts]

            if src_l != 0:
                l_size = len(l)
                if l_size > 1:
                    mixing[src_l] = -sum(
                        wgt * np.log2(wgt) for wgt in norm_wgts) / (l_size * np.log2(l_size)
                    )
                else:
                    mixing[src_l] = 0

        return mixing

    def compute_node_measures(
        self,
        net: nx.DiGraph,
        cores: list[set[Node]]=None,
    ) -> None:
        """Calculate node measures and save as attributes."""
        match self._config["sig_clu"]["scheme"]:
            case SigCluScheme.STANDARD:
                significant_nodes = set.union(*cores)
                for node in net.nodes():
                    net.nodes[node]["core"] = int(node in significant_nodes)
            case SigCluScheme.RECURSIVE:
                sorted_cores = sorted(cores, key=len, reverse=True)
                node_to_core = {}
                for index, node_set in enumerate(sorted_cores, start=1):
                    for node in node_set:
                        node_to_core[node] = index
                for node in net.nodes:
                    net.nodes[node]["core"] = node_to_core.get(node, 0)
            case _:
                pass

        # In- and out-degree and strength
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

        # Set node attributes
        if node_info:
            node_info = im.get_dataframe(["name", "module_id", "flow", "modular_centrality"])
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
        return bootstraps

    def sig_cluster(
        self,
        partition: Partition,
        bootstraps: tuple[Partition],
    ) -> list[set[Node]]:
        """Finds significant core(s) of each module in the partition."""
        cfg = self._config["sig_clu"]
        sig_clu = SigClu(partition, bootstraps, cfg)
        match cfg["scheme"]:
            case SigCluScheme.STANDARD:
                cores = sig_clu.run()
            case SigCluScheme.RECURSIVE:
                cores = []

                # Initialize
                nodes = set.union(*partition)
                core = nodes

                # Loop to find each core above min size threshold
                thresh = cfg["thresh"] * len(nodes)
                while True:
                    sig_clu.partition = [nodes]
                    core = sig_clu.run()[0]
                    if len(core) >= thresh:
                        cores.append(core)
                        nodes.difference_update(core)  # Remove nodes in core
                    else:
                        break
            case SigCluScheme.NONE:
                return []
        return cores

    def group_nodes_by_attr(self, net: nx.DiGraph, attr: str="module") -> Partition:
        """Groups nodes in a network into sets by their module."""
        node_attr = list(net.nodes(data=attr))
        df = pd.DataFrame(node_attr, columns=["node", attr])
        return df.groupby(attr)["node"].apply(set).tolist()

    def get_num_labels(self, net: nx.DiGraph, attr: str="module") -> int:
        """Get the number of labels in a network."""
        attrs = {net.nodes[node][attr] for node in net.nodes}
        return len(attrs)
