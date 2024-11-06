"""Command line interface."""
import click

from .cli import commands


@click.group()
def netclop():
    """Network clustering operations."""
    pass


netclop.add_command(commands.rsc)
