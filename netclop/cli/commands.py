"""Commands for the CLI."""
import click
import numpy as np

from ..config_loader import load_config, update_config
from ..networkops import NetworkOps
from ..plot import GeoPlot
from ..sigcore import SigCluScheme
from . import gui
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


@click.command(name="construct")
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


@click.command(name="partition")
@options.io
@options.binning
@options.comm_detection
@options.sig_clu
@click.option(
    "--node-centrality/--no-node-centrality",
    "calc_node_centrality",
    is_flag=True,
    show_default=True,
    default=True,
    help="Calculate node centrality indices.",
)
@click.option(
    "--plot/--no-plot",
    "do_plot",
    is_flag=True,
    show_default=True,
    default=True,
    help="Display geographic plot.",
)
@click.pass_context
def run(
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
    calc_node_centrality,
    do_plot,
):
    """Community detection and significance clustering."""
    gui.header("NETwork CLustering OPerations")
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
            "scheme": SigCluScheme[sc_scheme],
            },
        }
    update_config(ctx.obj["cfg"], updated_cfg)
    cfg = ctx.obj["cfg"]
    nops = NetworkOps(cfg)

    gui.subheader("Network construction")
    net = nops.from_positions(input_path)
    gui.info("Nodes", len(net.nodes))
    gui.info("Links", len(net.edges))

    gui.subheader("Community detection")
    nops.partition(net, set_node_attr=calc_node_centrality)
    gui.info("Modules", nops.get_num_labels(net, "module"))

    match cfg["sig_clu"]["scheme"]:
        case SigCluScheme.STANDARD | SigCluScheme.RECURSIVE:
            gui.subheader("Network perturbation")

            bs_nets = nops.make_bootstraps(net)
            gui.info("Resampled nets", len(bs_nets))

            for bs in bs_nets:
                nops.partition(bs, set_node_attr=False)

            part = nops.group_nodes_by_attr(net, "module")
            bs_parts = [nops.group_nodes_by_attr(bs, "module") for bs in bs_nets]

            counts = [len(bs_part) for bs_part in bs_parts]
            gui.info("Modules", f"{np.mean(counts):.1f} ± {np.std(counts):.1f}")

            gui.subheader("Significance clustering")
            cores = nops.sig_cluster(part, bs_parts)
            gui.info("Cores", len(cores))

            nops.associate_node_assignments(net, cores)
        case SigCluScheme.NONE:
            pass

    if calc_node_centrality:
        nops.calc_node_centrality(net)

    df = nops.to_dataframe(net, output_path)

    if do_plot:
        gplt = GeoPlot.from_dataframe(df)
        gplt.plot_structure(sc_scheme=cfg["sig_clu"]["scheme"])
        gplt.show()

    gui.footer()


@netclop.command(name="plot")
@options.io
@options.plot
@options.sc_scheme
@click.pass_context
def plot_structure(ctx, input_path, output_path, sc_scheme):
    """Plots structure."""
    sc_scheme = SigCluScheme[sc_scheme]

    gplt = GeoPlot.from_file(input_path)
    gplt.plot_structure(sc_scheme=sc_scheme)

    if output_path is not None:
        gplt.save(output_path)
    else:
        gplt.show()

@click.command(name="structure")
@options.io
@options.plot
@options.sc_scheme
@click.pass_context
def plot_structure(ctx, input_path, output_path, sc_scheme):
    """Plots network structure."""
    sc_scheme = SigCluScheme[sc_scheme]

    gplt = GeoPlot.from_file(input_path)
    gplt.plot_structure(sc_scheme=sc_scheme)

    if output_path is not None:
        gplt.save(output_path)
    else:
        gplt.show()

@click.command(name="centrality")
@options.io
@options.plot
@click.pass_context
def plot_centrality(ctx, input_path, output_path):
    """Plots node centrality indices."""
    gplt = GeoPlot.from_file(input_path)
    gplt.plot_centrality()

    if output_path is not None:
        gplt.save(output_path)
    else:
        gplt.show()