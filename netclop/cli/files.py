"""Handles output files."""
from datetime import datetime
from pathlib import Path


def make_run_id(prefix):
    """Make ID number for netclop run."""
    return str(prefix) + "_" + str(int(datetime.now().timestamp()))


def make_filepath(path: Path, field: str = None, extension: str = "png"):
    """Add a field to path."""
    if field is None:
        delimiter = ""
        field = ""
    else:
        delimiter = "_"
    return path.with_name(path.name + delimiter + field + "." + extension)
