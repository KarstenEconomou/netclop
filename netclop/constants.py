"""Defines constants."""
import os
import typing

type Node = str
type Partition = typing.Sequence[set[Node]]
type Path = typing.Union[str, os.PathLike]
