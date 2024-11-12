"""Command line interface."""
from netclop.cli.commands import *


@click.group()
def netclop():
    """Network clustering operations."""
    pass


netclop.add_command(rsc)
