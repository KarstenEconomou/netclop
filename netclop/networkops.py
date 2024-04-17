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

    def get_module_list(self, net: nx.DiGraph) -> set:
        """Gets set of module indices."""
        return {net.nodes[node]["module"] for node in net.nodes}

    def modular_strength(self, net: nx.DiGraph) -> dict:
        """Calculates strength measures of each module."""
        modules = self.get_module_list(net)
        strength = {
            module: {"ext": {"out": 0, "in": 0}, "int": 0} for module in modules
        }

        for src, tgt, wgt in net.edges.data("weight"):
            # Out-edge of src, in-edge of tgt, with weight wgt
            src_mod = net.nodes[src]["module"]
            tgt_mod = net.nodes[tgt]["module"]
            if src_mod != tgt_mod:
                strength[src_mod]["ext"]["out"] += wgt
                strength[tgt_mod]["ext"]["in"] += wgt
            else:
                strength[src_mod]["int"] += wgt

        return strength

    def coherence_fortress(self, strength: dict) -> tuple[dict[int, float], dict[int, float]]:
        """Computes the coherence and fortress of each module."""
        coherence = {}
        fortress = {}
        for mod in strength.keys():
            out_str = strength[mod]["int"] + strength[mod]["ext"]["out"]
            in_str = strength[mod]["int"] + strength[mod]["ext"]["in"]

            coherence[mod] = strength[mod]["int"] / out_str if out_str != 0 else 0
            fortress[mod] = strength[mod]["int"] / in_str if in_str != 0 else 0
        return coherence, fortress

    def cohesion_mixing(self, net: nx.DiGraph) -> tuple[float, float]:
        """Calculates cohesion and mixing of a partition."""
        strength = self.modular_strength(net)
        modules = strength.keys()

        net_str = sum(strength[mod]["int"] + strength[mod]["ext"]["out"] for mod in modules)
        net_int_str = sum(strength[mod]["int"] for mod in modules)

        cohesion = net_int_str / net_str

        module_mixing = self.mixing(net)
        mixing = sum(module_mixing[mod] * strength[mod]["int"] for mod in modules) / net_int_str

        return cohesion, mixing

    def mixing(self, net: nx.DiGraph) -> float:
        """Calculates the mixing parameter of each module."""
        mixing = {}
        modules = self.group_nodes_by_module(net)
        for mod in modules:
            retain_wgts = []
            for src in mod:
                src_mod = net.nodes[src]["module"]
                for tgt in net.successors(src):
                    tgt_mod = net.nodes[tgt]["module"]
                    if src_mod == tgt_mod:
                        wgt = net.get_edge_data(src, tgt)["weight"]
                        retain_wgts.append(wgt)

            total_retain_wgt = sum(retain_wgts)
            norm_wgts = [wgt / total_retain_wgt for wgt in retain_wgts]

            mod_size = len(mod)
            if mod_size > 1:
                mixing[src_mod] = -sum(
                    wgt * np.log2(wgt) for wgt in norm_wgts
                ) / (mod_size * np.log2(mod_size))
            else:
                mixing[src_mod] = 0

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

        # # In- and out-degree and strength
        # in_degs = dict(net.in_degree())
        # out_degs = dict(net.out_degree())

        # in_strs = dict(net.in_degree(weight='weight'))
        # out_strs = dict(net.out_degree(weight='weight'))

        # # Betweenness
        # betweenness_centrality = nx.betweenness_centrality(net)

        # # Update network with metrics
        # nx.set_node_attributes(net, in_degs, "in_deg")
        # nx.set_node_attributes(net, out_degs, "out_deg")
        # nx.set_node_attributes(net, in_strs, "in_str")
        # nx.set_node_attributes(net, out_strs, "out_str")
        # nx.set_node_attributes(net, betweenness_centrality, "betweenness")

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
                for module in partition:
                    thresh = cfg["thresh"] * len(module)

                    sig_clu.partition = [module]
                    core = sig_clu.run()[0]
                    noise = module.difference(core)

                    module_cores = []
                    while len(core) >= thresh:
                        module_cores.append(core)
                        sig_clu.partition = [noise]
                        core = sig_clu.run()[0]
                        noise = noise.difference(core)
                    cores.extend(module_cores)
            case SigCluScheme.NONE:
                return []
        return cores

    def group_nodes_by_module(self, net: nx.DiGraph) -> Partition:
        """Groups nodes in a network into sets by their module."""
        node_modules = list(net.nodes(data="module"))
        df = pd.DataFrame(node_modules, columns=["node", "module"])
        return df.groupby("module")["node"].apply(set).tolist()

    def get_num_modules(self, net: nx.DiGraph) -> int:
        """Get the number of modules in a network."""
        modules = {net.nodes[node]["module"] for node in net.nodes}
        return len(modules)
