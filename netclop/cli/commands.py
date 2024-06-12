"""Commands for the CLI."""
import enum
from pathlib import Path

import click

from .utils import InputData, read_net
from ..config_loader import load_config, update_config
from ..networkops import NetworkOps
from ..plot import GeoPlot
from ..sigcore import SigCluScheme
from . import gui

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
@click.option(
    "--res",
    type=click.IntRange(min=0, max=15),
    default=DEF_CFG["binning"]["res"],
    show_default=True,
    help="H3 grid resolution for domain discretization.",
)
@click.pass_context
def construct(ctx, input_path, output_path, res):
    """Constructs a network from LPT positions."""
    updated_cfg = {"binning": {"res": res}}
    update_config(ctx.obj["cfg"], updated_cfg)

    nops = NetworkOps(ctx.obj["cfg"])
    net = nops.net_from_positions(input_path)

    if output_path is not None:
        nops.write_edgelist(net, output_path)


@click.command(name="run")
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
    help="Output path where the nodelist will be saved.",
)
@click.option(
    "--input-data",
    "-i",
    "input_type",
    type=click.Choice([intype.name for intype in InputData], case_sensitive=False),
    default="LPT",
    show_default=True,
    help="Input data type: LPT for start/end position pairs, NET for weighted edgelist.",
)
@click.option(
    "--significance-cluster",
    "-sc",
    "sc_scheme",
    type=click.Choice([scheme.name for scheme in SigCluScheme], case_sensitive=False),
    default="NONE",
    show_default=True,
    help="Scheme to demarcate significant community assignments from statistical noise.",
)
@click.option(
    "--res",
    type=click.IntRange(min=0, max=15),
    default=DEF_CFG["binning"]["res"],
    show_default=True,
    help="H3 grid resolution for domain discretization.",
)
@click.option(
    "--markov-time",
    "-mt",
    type=click.FloatRange(min=0, max=None, min_open=True),
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
    "--num-trials",
    "-n",
    show_default=True,
    default=DEF_CFG["infomap"]["num_trials"],
    help="Number of outer-loop community detection trials to run.",
)
@click.option(
    "--seed",
    "-s",
    show_default=True,
    type=click.IntRange(min=1, max=None),
    default=DEF_CFG["infomap"]["seed"],
    help="PRNG seed for community detection.",
)
@click.option(
    "--cooling-rate",
    "-cr",
    "cool_rate",
    type=float,
    show_default=True,
    default=DEF_CFG["sig_clu"]["cool_rate"],
    help="Cooling rate in simulated annealing.",
)
@click.option(
    "--plot/--no-plot",
    "do_plot",
    is_flag=True,
    show_default=True,
    default=True,
    help="Display geographic plot.",
)
@click.option(
    "--mute-trivial/--show-trivial",
    "mute_trivial",
    is_flag=True,
    show_default=True,
    default=True,
    help="Grey trivial modules.",
)
@click.pass_context
def run(
    ctx,
    input_path,
    output_path,
    input_type,
    sc_scheme,
    res,
    markov_time,
    variable_markov_time,
    num_trials,
    seed,
    cool_rate,
    do_plot,
    mute_trivial,
):
    """Community detection and significance clustering."""
    input_path = Path(input_path)
    input_type = InputData[input_type]
    sc_scheme = SigCluScheme[sc_scheme]
    if input_path.is_dir():  # Necessitate ensemble input to use recursive sc
        sc_scheme = SigCluScheme.RECURSIVE

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
            "scheme": sc_scheme,
            },
        }
    update_config(ctx.obj["cfg"], updated_cfg)
    cfg = ctx.obj["cfg"]
    nops = NetworkOps(cfg)

    gui.subheader("Network construction")

    if input_path.is_file():
        net = read_net(nops, input_path, input_type)

        gui.info("Nodes", len(net.nodes))
        gui.info("Links", len(net.edges))

        nops.calc_node_centrality(net)

        gui.subheader("Community detection")
        nops.partition(net)
        gui.info("Modules", nops.get_num_labels(net, "module"))
    elif input_path.is_dir():
        bs_nets = [read_net(nops, path, input_type) for path in input_path.glob('*.csv')]
        gui.info("Replicate nets", len(bs_nets))
        gui.report_average("Nodes", [len(bs_net.nodes) for bs_net in bs_nets])
        gui.report_average("Edges", [len(bs_net.edges) for bs_net in bs_nets])

        net = bs_nets[0]

    match cfg["sig_clu"]["scheme"]:
        case SigCluScheme.STANDARD | SigCluScheme.RECURSIVE:
            gui.subheader("Network ensemble")

            if input_path.is_file():
                bs_nets = nops.make_bootstraps(net)
                gui.info("Resampled nets", len(bs_nets))
                part = nops.group_nodes_by_attr(net, "module")
            elif input_path.is_dir():
                part = [set().union(*[set(bs.nodes) for bs in bs_nets])]  # Dummy reference partition

            bs_parts = []
            for bs in bs_nets:
                nops.partition(bs, set_node_attr=False)
                bs_parts.append(nops.group_nodes_by_attr(bs, "module"))

            gui.report_average("Modules", [len(bs_part) for bs_part in bs_parts])

            gui.subheader("Significance clustering")
            cores = nops.sig_cluster(part, bs_parts)
            gui.info("Cores", len(cores))

            nops.associate_node_assignments(net, cores)
        case SigCluScheme.NONE:
            pass

    df = nops.to_dataframe(net, output_path)

    if do_plot:
        gplt = GeoPlot.from_dataframe(df)
        gplt.plot_structure(sc_scheme=cfg["sig_clu"]["scheme"], mute_trivial=mute_trivial)
        gplt.show()

    gui.footer()


@netclop.command(name="structure")
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
@click.option(
    "--significance-cluster",
    "-sc",
    "sc_scheme",
    type=click.Choice([scheme.name for scheme in SigCluScheme], case_sensitive=False),
    default="NONE",
    show_default=True,
    help="Scheme to demarcate significant community assignments from statistical noise.",
)
@click.option(
    "--mute-trivial/--show-trivial",
    "mute_trivial",
    is_flag=True,
    show_default=True,
    default=True,
    help="Grey trivial modules.",
)
@click.pass_context
def plot_structure(ctx, input_path, output_path, sc_scheme, mute_trivial):
    """Plots structure."""
    sc_scheme = SigCluScheme[sc_scheme]

    gplt = GeoPlot.from_file(input_path)
    gplt.plot_structure(sc_scheme=sc_scheme, mute_trivial=mute_trivial)

    if output_path is not None:
        gplt.save(output_path)
    else:
        gplt.show()

@click.command(name="centrality")
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
@click.pass_context
def plot_centrality(ctx, input_path, output_path):
    """Plots node centrality indices."""
    gplt = GeoPlot.from_file(input_path)
    gplt.plot_centrality()

    if output_path is not None:
        gplt.save(output_path)
    else:
        gplt.show()