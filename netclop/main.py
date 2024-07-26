"""Command line interface."""
import click

from .cli import commands
from .config_loader import load_config


@click.group(invoke_without_command=True, name="netclop")
@click.option(
    '--config', 
    "config_path",
    type=click.Path(exists=True),
    help="Path to custom configuration file.",
)
@click.pass_context
def netclop(ctx, config_path):
    """Network clustering operations."""
    if ctx.obj is None:
        ctx.obj = {}
    cfg = load_config()
    if config_path:
        cfg.update(load_config(config_path))
    ctx.obj["cfg"] = cfg

@netclop.group()
def plot():
    """Spatially-embedded network plotting."""
    pass


netclop.add_command(commands.construct)
netclop.add_command(commands.rsc)

plot.add_command(commands.plot_structure)
plot.add_command(commands.plot_centrality)