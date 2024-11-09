"""Defines types."""

type Cell = int
type Node = str
type NodeSet = set[Node] | frozenset[Node]
type Partition = list[NodeSet]
type CentralityNodes = dict[Node, float]
