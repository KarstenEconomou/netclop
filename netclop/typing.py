"""Defines types."""
type Cell = int
type NodeMetric = dict[Node, float | int]
type Node = str
type NodeSet = set[Node] | frozenset[Node]
type Partition = list[NodeSet]
