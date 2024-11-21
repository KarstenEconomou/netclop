"""Node centrality computations."""
import networkx as nx

from netclop.constants import WEIGHT_ATTR
from netclop.typing import NodeMetric


def out_strength(net: nx.DiGraph, **kwargs) -> NodeMetric:
    """Compute the out-strength of nodes."""
    return dict((node, out_str) for node, out_str in net.out_degree(weight=WEIGHT_ATTR))


def in_strength(net: nx.DiGraph, **kwargs) -> NodeMetric:
    """Compute the in-strength of nodes."""
    return dict((node, in_str) for node, in_str in net.in_degree(weight=WEIGHT_ATTR))


def excess(net: nx.DiGraph, **kwargs) -> NodeMetric:
    """Compute the in-strength minus out-strength of nodes."""
    out_str = out_strength(net, **kwargs)
    in_str = in_strength(net, **kwargs)
    return dict((n, i - o) for n, o, i in zip(out_str.keys(), out_str.values(), in_str.values()))
