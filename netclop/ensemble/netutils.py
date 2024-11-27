"""Network utility functions."""
from typing import Sequence

from netclop.exceptions import OverlappingPartitionError
from netclop.typing import NodeSet, Partition, NodeMetric


def flatten_partition(partition: Partition | Sequence[Partition]) -> NodeSet:
    """Flattens a partition to the set of elements partitioned."""
    if isinstance(partition, Sequence) and not isinstance(partition[0], set | frozenset):
        return flatten_partition([flatten_partition(part) for part in partition])
    return frozenset().union(*partition)


def label_partition(partition: Partition) -> NodeMetric:
    """Creates labels for a partition."""
    labels = {}
    labelled = []

    for label, part in enumerate(partition, 1):
        for node in part:
            if node in labelled:
                raise OverlappingPartitionError

            labels[node] = label
            labelled.append(node)
    return labels
