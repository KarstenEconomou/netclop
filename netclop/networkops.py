"""Defines the NetworkOps class."""
import dataclasses
import typing

import click
import h3.api.numpy_int as h3
import networkx as nx
import numpy as np
import pandas as pd
from infomap import Infomap

from .constants import *
from .sigcore import SigClu

@dataclasses.dataclass
class NetworkOps:
    """Network operations."""
    network: nx.DiGraph

    part_config: dict[str, any] = dataclasses.field(default_factory=lambda: DEFAULT_PART_CONFIG)
    bs_config: dict[str, any] = dataclasses.field(default_factory=lambda: DEFAULT_BS_CONFIG)
    sc_config: dict[str, any] = dataclasses.field(default_factory=lambda: DEFAULT_SC_CONFIG)

    def edge_list_to_file(self, path: click.Path) -> None:
        """Writes the network edge list to file."""
        nx.write_edgelist(
            self.network,
            path,
            delimiter=",",
            comments="#",
            data=["wgt", "wgt_nrm"],
        )

    def partition(self, net: nx.DiGraph = None, path: click.Path = None, silent: bool = False) -> pd.DataFrame:
        """Partitions a network and saves its modular description."""
        if net is None:
            net = self.network

        im = Infomap(
            silent=True,
            two_level=True,
            flow_model="directed",
            num_trials=self.part_config["num_trials"],
            markov_time=self.part_config["markov_time"],
            seed=self.part_config["seed"],
        )
        _ = im.add_networkx_graph(net, weight="wgt")  # Requires node "names" to be strings
        im.run()

        node_info = im.get_dataframe(["name", "module_id", "flow", "modular_centrality"])
        node_info = node_info.rename(columns={"name": "node", "module_id": "module"})

        if path is not None:
            node_info.to_csv(path, index=False)

        if not silent:
            click.echo(f"Partitioned network into {len(node_info["module"].unique())} modules")

        return node_info

    def gen_perturb_nets(self) -> tuple[nx.DiGraph, ...]:
        """Resample out-edge weight distributions to perturb networks."""
        net = self.network
        size = self.bs_config["size"]
        tuning_param = self.bs_config["tuning_param"]
        rng = np.random.default_rng(self.bs_config["seed"])

        pert_nets = [net.copy() for _ in range(size)]

        for u in net.nodes:
            # Resample out-edge weight distribution of node
            conc = tuning_param * np.fromiter((w for _, _, w in net.out_edges(u, data="wgt_nrm")), dtype=float)
            realizations = rng.dirichlet(conc, size=size)

            # Assign edge weights
            for pert_net, realization in zip(pert_nets, realizations):
                for (_, v), wgt in zip(net.out_edges(u), realization):
                    pert_net[u][v]["wgt_nrm"] = wgt

        click.echo(f"Generated {size} perturbed networks")
        return pert_nets

    def significance_cluster(self, path: click.Path = None):
        """Performs significance clustering."""
        def get_module_node_map(node_info: pd.DataFrame) -> Partition:
            """Create set-of-sets partition representation from node-module pairs."""
            return tuple(node_info.groupby("module")["node"].apply(set))

        node_info = self.partition()
        clus = get_module_node_map(node_info)

        pert_nets = self.gen_perturb_nets()
        pert_clus = tuple(
            get_module_node_map(self.partition(pert_net, silent=True)) for pert_net in pert_nets
        )

        mean_clu_size = np.mean([len(clu) for clu in pert_clus])
        std_clu_size = np.std([len(clu) for clu in pert_clus])
        click.echo(f"Partitioned perturbed networks into {mean_clu_size}+/-{std_clu_size} modules")

        sig_clu = SigClu(clus, pert_clus, config=self.sc_config)
        cores = []
        for clu in clus:
            core = sig_clu.find_sig_core(clu)
            cores.append(core)

        significant_nodes = set().union(*cores)
        node_info["significant"] = node_info["node"].apply(
            lambda x: 1 if x in significant_nodes else 0
        )
        if path is not None:
            node_info.to_csv(path, index=False)

    @classmethod
    def from_locations(cls, path: click.Path, res: int, *args, **kwargs) -> typing.Self:
        """Constructs a network from file of initial and final coordinates."""
        data = pd.read_csv(
            path,
            names=["initial_lng", "initial_lat", "final_lng", "final_lat"],
            index_col=False,
            comment="#",
        )

        click.echo(f"Binning {data.shape[0]} particle positions")
        srcs = cls.bin_positions(data["initial_lng"], data["initial_lat"], res)
        tgts = cls.bin_positions(data["final_lng"], data["final_lat"], res)
        edges = tuple(zip(srcs, tgts))
        return cls(cls.construct_net(edges), *args, **kwargs)

    @classmethod
    def from_file(cls, path: click.Path, *args, **kwargs) -> typing.Self:
        """Constructs a network from edge list file."""
        net = nx.read_edgelist(
            path,
            comments="#",
            delimiter=",",
            create_using=nx.DiGraph,
            nodetype=str,
            data=[("wgt", float), ("wgt_nrm", float)],
        )
        return cls(net, *args, **kwargs)

    @staticmethod
    def bin_positions(
        lngs: typing.Sequence[float],
        lats: typing.Sequence[float],
        res: int,
    ) -> list[Node]:
        """Bins (lng, lat) coordinate pairs into an H3 cell."""
        bins = [h3.latlng_to_cell(lat, lng, res) for lat, lng in zip(lats, lngs)]
        return bins

    @staticmethod
    def construct_net(edges: typing.Sequence[tuple[Node, Node]]) -> nx.Graph:
        """Constructs a network from edge list."""
        net = nx.DiGraph()
        for src, tgt in edges:
            if net.has_edge(src, tgt):
                # Record another transition along a recorded edge
                net[src][tgt]["wgt"] += 1
            else:
                # Record a new edge
                net.add_edge(src, tgt, wgt=1)

        for src in net.nodes:
            out_wgt = sum(wgt for _, _, wgt in net.out_edges(src, data='wgt', default=0))
            for tgt in net.successors(src):
                net[src][tgt]["wgt_nrm"] = net[src][tgt]["wgt"] / out_wgt if out_wgt != 0 else 0

        nx.relabel_nodes(net, dict((name, str(name)) for name in net.nodes), copy=False)
        click.echo(f"Constructed network of {len(net.nodes)} nodes and {len(net.edges)} edges")
        return net
