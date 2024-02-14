"""Command line interface."""
import functools

import click
import numpy as np

from .config_loader import load_config, update_config
from .networkops import NetworkOps
from .plot import GeoPlot

DEF_CFG = load_config()


@click.group(invoke_without_command=True)
@click.option(
    '--config', 
    "config_path",
    type=click.Path(exists=True),
    help="Path to custom configuration file.",
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
        required=False,
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
    updated_cfg = {"binning": {"res": res}}
    update_config(ctx.obj["cfg"], updated_cfg)

    netops = NetworkOps(ctx.obj["cfg"])
    net = netops.from_positions(input_path)
    netops.write_edgelist(net, output_path)


@netclop.command(name="partition")
@path_options
@click.option(
    "--significance-cluster", 
    "-sc",
    "sig_clu",
    is_flag=True,
    help="Demarcate significant community assignments from statistical noise.",
)
@click.option(
    "--res",
    type=int,
    default=DEF_CFG["binning"]["res"],
    show_default=True,
    help="H3 grid resolution (0-15) for domain discretization.",
)
@click.option(
    "--markov-time",
    "-mt",
    type=float,
    default=DEF_CFG["infomap"]["markov_time"],
    show_default=True,
    help="Markov time to tune spatial scale of detected structure.",
)
@click.option(
    "--variable-markov-time/--static-markov-time",
    is_flag=True,
    show_default=True,
    default=DEF_CFG["infomap"]["variable_markov_time"],
    help="Permits the dynamic adjustment of Markov time with varying density.",
)
@click.option(
    "--cooling-rate",
    "-cr",
    "cool_rate",
    show_default=True,
    default=DEF_CFG["sig_clu"]["cool_rate"],
    help="Cooling rate in simulated annealing.",
)
@click.pass_context
def partition(
    ctx,
    input_path,
    output_path,
    sig_clu,
    res,
    markov_time,
    variable_markov_time,
    cool_rate,
):
    """Runs significance clustering directly from LPT positions."""
    updated_cfg = {
        "binning": {
            "res": res
            },
        "infomap": {
            "markov_time": markov_time,
            "variable_markov_time": variable_markov_time,
            },
        "sig_clu": {
            "cool_rate": cool_rate,
            },
        }
    update_config(ctx.obj["cfg"], updated_cfg)

    netops = NetworkOps(ctx.obj["cfg"])

    net = netops.from_positions(input_path)
    netops.partition(net)

    if sig_clu:
        bootstrap_nets = netops.make_bootstraps(net)
        for bootstrap in bootstrap_nets:
            netops.partition(bootstrap, node_info=False)

        part = netops.group_nodes_by_module(net)
        bootstrap_parts = [netops.group_nodes_by_module(bs_net) for bs_net in bootstrap_nets]

        counts = [len(bs_part) for bs_part in bootstrap_parts]
        print(f"Partitioned into {np.mean(counts):.1f} +/- {np.std(counts):.1f} modules")

        cores = netops.significance_cluster(part, bootstrap_parts)
        netops.compute_node_measures(net, cores)

    df = netops.to_dataframe(net, output_path)
    gplt = GeoPlot.from_dataframe(df)
    gplt.plot(delineate_noise=sig_clu)


@netclop.command(name="plot")
@click.argument(
    "input-path", 
    type=click.Path(exists=True),
    required=True,
    )
@click.option(
    "--delineate-noise",
    "-dn",
    is_flag=True,
    help="Demarcate significant community assignments from statistical noise.",
)
def plot(input_path, delineate_noise):
    """Plots nodes."""
    gplt = GeoPlot.from_file(input_path)
    gplt.plot(delineate_noise=delineate_noise)

if __name__ == '__main__':
    netclop(obj={})
