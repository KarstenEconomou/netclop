from dataclasses import dataclass
from os import PathLike
from typing import Sequence

import networkx as nx
import pandas as pd
from h3.api import numpy_int as h3

from netclop.constants import WEIGHT_ATTR
from netclop.typing import Cell


class GeoNet:
    """Helper class for network construction from geographic data."""
    @dataclass(frozen=True)
    class Config:
        res: int = 5

    def __init__(self, **config_options):
        self.cfg = self.Config(**config_options)

    def bin_positions(self, lngs: Sequence[float], lats: Sequence[float]) -> list[Cell]:
        """Bin (lng, lat) coordinate pairs into an H3 cell."""
        return [h3.latlng_to_cell(lat, lng, self.cfg.res) for lat, lng in zip(lats, lngs)]

    def make_lpt_edges(self, path: PathLike) -> tuple[tuple[Cell, Cell], ...]:
        """Make an edge list (with duplicates) from LPT positions."""
        data = pd.read_csv(
            path,
            names=["initial_lng", "initial_lat", "final_lng", "final_lat"],
            index_col=False,
            comment="#",
        )

        srcs = self.bin_positions(data["initial_lng"], data["initial_lat"])
        tgts = self.bin_positions(data["final_lng"], data["final_lat"])
        return tuple(zip(srcs, tgts))

    def net_from_lpt(self, path: PathLike) -> nx.DiGraph:
        """Construct a network from LPT positions."""
        net = nx.DiGraph()
        edges = self.make_lpt_edges(path)

        for src, tgt in edges:
            if net.has_edge(src, tgt):
                # Record another transition along a recorded edge
                net[src][tgt][WEIGHT_ATTR] += 1
            else:
                # Record a new edge
                net.add_edge(src, tgt, weight=1)

        nx.relabel_nodes(net, dict((name, str(name)) for name in net.nodes), copy=False)
        return net
