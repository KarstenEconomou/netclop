import enum
from pathlib import Path

from networkx import DiGraph

from ..networkops import NetworkOps


class InputData(enum.Enum):
    """Input data type."""
    LPT = enum.auto()
    NET = enum.auto()


def read_net(nops: NetworkOps, input_path: Path, input_type: InputData) -> DiGraph:
    """Reads a network from a file."""
    match input_type:
        case InputData.LPT:
            net = nops.net_from_positions(input_path)
        case InputData.NET:
            net = nops.net_from_file(input_path)
    return net

