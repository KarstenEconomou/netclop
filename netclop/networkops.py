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
    network: nx.DiGraph

    def to_file(self, path: click.Path) -> None:
        nx.write_edgelist(
            self.network,
            path,
            delimiter=",",
            comments="#",
            data=['wgt', 'wgt_nrm'],
        )

    def partition(
            self,
            path: click.Path,
            num_trials: int,
            markov_time: float,
            seed: int,
        ) -> None:
        im = Infomap(
            silent=True,
            two_level=True,
            flow_model="directed",
            num_trials=num_trials,
            markov_time=markov_time,
            seed=seed,
        )
        _ = im.add_networkx_graph(self.network, weight="wgt")
        im.run()

        partition = im.get_dataframe(["name", "module_id", "flow", "modular_centrality"])
        partition[["name", "module_id", "flow"]].to_csv(path, index=False, header=False)
        #tuple(partition.groupby("module_id")["name"].apply(set))

    @classmethod
    def from_locations(cls, path: click.Path, res: int) -> typing.Self:
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
        return NetworkOps(cls.construct_net(edges))
    
    @classmethod
    def from_file(cls, path: click.Path) -> typing.Self:
        net = nx.read_edgelist(
            path,
            comments="#",
            delimiter=",",
            create_using=nx.DiGraph,
            nodetype=str,
            data=[("wgt", float), ("wgt_nrm", float)],
        )
        return NetworkOps(net)

    @staticmethod
    def bin_positions(lngs: typing.Sequence[float], lats: typing.Sequence[float], res: int) -> list[Node]:
        bins = [h3.latlng_to_cell(lat, lng, res) for lat, lng in zip(lats, lngs)]
        return bins
    
    @staticmethod
    def construct_net(edges: typing.Sequence[tuple[Node, Node]]) -> nx.Graph:
        net = nx.DiGraph()
        for src, tgt in edges:
            if net.has_edge(src, tgt):
                # Record another transition along a recorded edge
                net[src][tgt]["wgt"] += 1
            else:
                # Record a new edge
                net.add_edge(src, tgt, wgt=1)

        for node in net.nodes:
            out_weight = sum(wgt for _, _, wgt in net.out_edges(node, data='wgt', default=0))
            for neighbour in net.successors(node):
                net[node][neighbour]["wgt_nrm"] = net[node][neighbour]["wgt"] / out_weight if out_weight != 0 else 0

        click.echo(f"Constructed network of {len(net.nodes)} nodes and {len(net.edges)} edges")
        return net
