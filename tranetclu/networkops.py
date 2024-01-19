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

    def to_file(self, path: str) -> None:
        nx.write_edgelist(self.network, path, delimiter=",", comments="#")

    def partition(
            self,
            file: str,
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
        _ = im.add_networkx_graph(self.network)
        im.run()

        partition = im.get_dataframe(["name", "module_id", "flow", "modular_centrality"])
        partition["name", "module_id", "flow"].to_csv(file)
        #tuple(partition.groupby("module_id")["name"].apply(set))

    @classmethod
    def from_locations(cls, path: str, res: int) -> typing.Self:
        data = pd.read_csv(
            path,
            usecols=["initial_lng", "initial_lat", "final_lng", "final_lat"],
            index_col=False,
            comment="#",
        )

        srcs = cls.bin_positions(data["initial_lng"], data["initial_lat"], res)
        tgts = cls.bin_positions(data["final_lng"], data["final_lat"], res)
        edges = tuple(zip(srcs, tgts))

        return NetworkOps(cls.construct_net(edges))
    
    @classmethod
    def from_file(cls, path: str) -> typing.Self:
        net = nx.read_edgelist(
            path, 
            comments="#", 
            delimiter=",",
            create_using=nx.DiGraph,
            nodetype=int,
            data=[("wgt", float), ("wgt_nrm", float)],
        )
        return NetworkOps(net)

    @staticmethod
    def bin_positions(lngs: typing.Sequence[float], lats: typing.Sequence[float], res: int) -> list[Node]:
        bins = [h3.latlng_to_cell(lat, lng, res) for lat, lng in zip(lats, lngs)]
        click.echo(f"Binned {len(lngs)} particle trajectories into {len(set(bins))} bins")
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
            out_weight = sum(net.out_edges(node, data='wgt', default=0).values())
            for neighbour in net.successors(node):
                net[node][neighbour]["wgt_nrm"] = net[node][neighbour]["wgt"] / out_weight if out_weight != 0 else 0

        click.echo(f"Constructed network of {len(net.nodes)} nodes and {len(net.edges)} edges")
        return net
