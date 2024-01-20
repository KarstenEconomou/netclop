import dataclasses
import typing

import click
import h3.api.numpy_int as h3
import networkx as nx
import pandas as pd
from infomap import Infomap

type Node = int
type Partition = tuple[set[Node], ...]

@dataclasses.dataclass
class NetworkOps:
    """Network operations."""
    network: nx.DiGraph
    modular: typing.Optional[pd.DataFrame] = None

    def to_file(self, path: click.Path) -> None:
        """Writes the network edge list to file."""
        nx.write_edgelist(
            self.network,
            path,
            delimiter=",",
            comments="#",
            data=['wgt', 'wgt_nrm'],
        )

    def partition(
            self,
            num_trials: int,
            markov_time: float,
            seed: int,
        ) -> None:
        """Partitions a network and saves its modular description."""
        im = Infomap(
            silent=True,
            two_level=True,
            flow_model="directed",
            num_trials=num_trials,
            markov_time=markov_time,
            seed=seed,
        )
        _ = im.add_networkx_graph(self.network, weight="wgt")  # Requires node "names" to be strings
        im.run()

        modular = im.get_dataframe(["name", "module_id", "flow", "modular_centrality"])
        self.modular = modular.rename(columns={"name": "node", "module_id": "module"})
        click.echo(f"Partitioned into {len(self.modular["module"].unique())} modules")

    def partition_to_file(self, path: click.Path) -> None:
        """Writes the network modular description to file."""
        self.modular[["node", "module", "flow"]].to_csv(path, index=False, header=False)

    @classmethod
    def from_locations(cls, path: click.Path, res: int) -> typing.Self:
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
        return cls(cls.construct_net(edges))

    @classmethod
    def from_file(cls, path: click.Path) -> typing.Self:
        """Constructs a network from edge list file."""
        net = nx.read_edgelist(
            path,
            comments="#",
            delimiter=",",
            create_using=nx.DiGraph,
            nodetype=str,
            data=[("wgt", float), ("wgt_nrm", float)],
        )
        return cls(net)

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
