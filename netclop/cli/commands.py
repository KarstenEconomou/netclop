"""Commands for the CLI."""
import click
import numpy as np

from ..config_loader import load_config, update_config
from ..networkops import NetworkOps
from ..plot import GeoPlot
from ..sigcore import SigCluScheme
from . import options

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


@netclop.command(name="construct")
@options.io
@options.binning
@click.pass_context
def construct(ctx, input_path, output_path, res):
    """Constructs a network from LPT positions."""
    updated_cfg = {"binning": {"res": res}}
    update_config(ctx.obj["cfg"], updated_cfg)

    nops = NetworkOps(ctx.obj["cfg"])
    net = nops.from_positions(input_path)

    if output_path is not None:
        nops.write_edgelist(net, output_path)


@netclop.command(name="partition")
@options.io
@options.binning
@options.comm_detection
@options.sig_clu
@click.option(
    "--plot/--no-plot",
    "do_plot",
    is_flag=True,
    show_default=True,
    default=True,
    help="Show geographic plot of community structure.",
)
@click.pass_context
def partition(
    ctx,
    input_path,
    output_path,
    sc_scheme,
    res,
    markov_time,
    variable_markov_time,
    num_trials,
    seed,
    cool_rate,
    do_plot,
):
    """Runs significance clustering directly from LPT positions."""
    updated_cfg = {
        "binning": {
            "res": res
            },
        "infomap": {
            "markov_time": markov_time,
            "variable_markov_time": variable_markov_time,
            "num_trials": num_trials,
            "seed": seed,
            },
        "sig_clu": {
            "cool_rate": cool_rate,
            },
        }
    update_config(ctx.obj["cfg"], updated_cfg)
    cfg = ctx.obj["cfg"]
    cfg["sig_clu"]["scheme"] = SigCluScheme[sc_scheme]
    nops = NetworkOps(cfg)

    net = nops.from_positions(input_path)
    print(f"Construction: {len(net.nodes)} nodes, {len(net.edges)} links")

    nops.partition(net)
    print(f"Partition: {nops.get_num_labels(net, "module")} modules")

    match cfg["sig_clu"]["scheme"]:
        case SigCluScheme.STANDARD | SigCluScheme.RECURSIVE:
            bootstrap_nets = nops.make_bootstraps(net)
            for bootstrap in bootstrap_nets:
                nops.partition(bootstrap, node_info=False)

            part = nops.group_nodes_by_attr(net, "module")
            bootstrap_parts = [nops.group_nodes_by_attr(bs_net, "module") for bs_net in bootstrap_nets]
            print(f"Bootstrap: Resampled {len(bootstrap_parts)} nets")

            counts = [len(bs_part) for bs_part in bootstrap_parts]
            print(f"Bootstrap: Partition into {np.mean(counts):.1f} +/- {np.std(counts):.1f} modules")

            cores = nops.sig_cluster(part, bootstrap_parts)

            nops.compute_node_measures(net, cores)

            print(nops.cohesion_mixing(net, "core"))
        case SigCluScheme.NONE:
            print(nops.cohesion_mixing(net, "module"))

    df = nops.to_dataframe(net, output_path)

    if do_plot:
        gplt = GeoPlot.from_dataframe(df)
        gplt.plot(sc_scheme=cfg["sig_clu"]["scheme"])
        gplt.show()


@netclop.command(name="plot")
@options.io
@options.plot
@options.sc_scheme
@click.pass_context
def plot(ctx, input_path, output_path, sc_scheme):
    """Plots nodes."""
    sc_scheme = SigCluScheme[sc_scheme]

    gplt = GeoPlot.from_file(input_path)
    gplt.plot(sc_scheme=sc_scheme)

    if output_path is not None:
        gplt.save(output_path)
    else:
        gplt.show()
