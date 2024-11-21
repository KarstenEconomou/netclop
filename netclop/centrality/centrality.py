"""Node centrality handling."""
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Callable

from netclop.centrality.centrality_compute import *
from netclop.typing import NodeMetric


class CentralityScale(Flag):
    """Enum for color scale of centrality."""
    SEQUENTIAL = auto()
    DIVERGING = auto()


@dataclass
class CentralityIndex:
    """Class to encapsulate centrality index."""
    compute: Callable[..., NodeMetric]
    scale: CentralityScale


@dataclass
class CentralityRegistry:
    """Class to behave as registry of CentralityIndex."""
    _registry_map: dict[str, CentralityIndex] = field(default_factory=dict)

    def __post_init__(self):
        self.register("out-degree", nx.out_degree_centrality, CentralityScale.SEQUENTIAL)
        self.register("in-degree", nx.in_degree_centrality, CentralityScale.SEQUENTIAL),
        self.register("out-strength", out_strength, CentralityScale.SEQUENTIAL),
        self.register("in-strength", in_strength, CentralityScale.SEQUENTIAL),
        self.register("betweenness", nx.betweenness_centrality, CentralityScale.SEQUENTIAL),
        self.register("pagerank", nx.pagerank, CentralityScale.SEQUENTIAL),
        self.register("excess", excess, CentralityScale.DIVERGING),

    def register(self, name: str, compute: Callable[..., NodeMetric], scale: CentralityScale) -> None:
        """Add CentralityIndex to the registry."""
        self._registry_map[name] = CentralityIndex(compute, scale)

    def get(self, name: str) -> CentralityIndex:
        """Get CentralityIndex from its name."""
        if name not in self._registry_map:
            raise ValueError(f"Metric '{name}' is not found.")
        return self._registry_map[name]

    @property
    def registered(self):
        """Get list of registered CentralityIndex."""
        return list(self._registry_map.keys())


centrality_registry: CentralityRegistry = CentralityRegistry()
