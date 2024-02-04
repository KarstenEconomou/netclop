"""Command line interface."""
import functools

import click
import numpy as np

from .config_loader import load_config
from .networkops import NetworkOps
from .plot import GeoPlot

DEF_CFG = load_config()


@click.group(invoke_without_command=True)
@click.option(
    '--config', 
    "config_path",
    type=click.Path(exists=True), 
    help="Path to custom configuration file."
)
@click.pass_context
def netclop(ctx, config_path):
    """Netclop CLI."""
    if ctx.obj is None:
        ctx.obj = {}
    cfg = load_config()
    if config_path:
        cfg.update(load_config(config_path))
    ctx.obj["cfg"] = cfg


def path_options(f):
    """Specify input and output arguments."""
    @click.argument(
    "input-path", 
    type=click.Path(exists=True),
    )
    @click.option(
        "--output", 
        "-o",
        "output_path", 
        type=click.Path(),
        required=True,
        help="Output file.",
    )
    @functools.wraps(f)
    def wrapper_path_options(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_path_options


@netclop.command(name="construct")
@path_options
@click.option(
    "--res",
    type=int,
    default=DEF_CFG["binning"]["res"],
    show_default=True,
    help="H3 grid resolution (0-15) for domain discretization.",
)
@click.pass_context
def construct_net(ctx, input_path, output_path, res):
    """Constructs a network from LPT positions."""
    netops = NetworkOps(ctx.obj["cfg"])
    net = netops.from_positions(input_path)
    netops.write_edgelist(net, output_path)


@netclop.command(name="stream")
@path_options
@click.pass_context
def stream(ctx, input_path, output_path):
    """Runs significance clustering directly from LPT positions."""
    netops = NetworkOps(ctx.obj["cfg"])

    net = netops.from_positions(input_path)
    netops.partition(net)

    bootstrap_nets = netops.make_bootstraps(net)
    for bootstrap in bootstrap_nets:
        netops.partition(bootstrap, node_info=False)

    partition = netops.group_nodes_by_module(net)
    bootstrap_partitions = [netops.group_nodes_by_module(bs_net) for bs_net in bootstrap_nets]

    counts = [len(bs_part) for bs_part in bootstrap_partitions]
    print(f"Partitioned into {np.mean(counts):.1f} +/- {np.std(counts):.1f} modules")

    cores = netops.significance_cluster(partition, bootstrap_partitions)

    netops.compute_node_measures(net, cores)
    netops.write_nodelist(net, output_path)

    gplt = GeoPlot.from_file(output_path)
    gplt.plot()


@netclop.command(name="plot")
@click.argument(
    "input-path", 
    type=click.Path(exists=True),
    required=True,
    )
def plot(input_path):
    """Plots nodes."""
    gplt = GeoPlot.from_file(input_path)
    gplt.plot()

if __name__ == '__main__':
    netclop(obj={})
