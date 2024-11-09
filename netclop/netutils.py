"""Network utility functions."""
from typing import Sequence

from .typing import Node, Partition


def flatten_partition(partition: Partition | Sequence[Partition]) -> frozenset[Node]:
    if isinstance(partition, Sequence) and not isinstance(partition[0], set | frozenset):
        return flatten_partition([flatten_partition(part) for part in partition])
    return frozenset().union(*partition)
