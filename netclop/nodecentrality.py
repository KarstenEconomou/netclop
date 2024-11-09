"""Node centrality computations."""
import networkx as nx

from .constants import WEIGHT_ATTR
from .typing import Node

def out_strength(net: nx.DiGraph, **kwargs) -> dict[Node, float]:
    """Compute the out-strength of nodes."""
    return dict((node, out_str) for node, out_str in net.out_degree(weight=WEIGHT_ATTR))

def in_strength(net: nx.DiGraph, **kwargs) -> dict[Node, float]:
    """Compute the in-strength of nodes."""
    return dict((node, in_str) for node, in_str in net.in_degree(weight=WEIGHT_ATTR))