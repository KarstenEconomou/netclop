"""Commands for the CLI."""
import enum
from pathlib import Path

import click

from .utils import InputData, read_net
from ..config_loader import load_config, update_config
from ..networkops import NetworkOps
from ..plot import GeoPlot, UpSetPlot
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
    "output_dir",
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


@click.command(name="rsc")
@click.argument(
    "input-path",
    type=click.Path(exists=True),
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help="Output directory.",
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
    "--sig",
    "sig",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    show_default=True,
    default=DEF_CFG["sig_clu"]["sig"],
    help="Significance level for significance clustering.",
)
@click.option(
    "--cooling-rate",
    "-cr",
    "cool_rate",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    show_default=True,
    default=DEF_CFG["sig_clu"]["cool_rate"],
    help="Cooling rate in simulated annealing.",
)
@click.option(
    "--size-thresh",
    "thresh",
    type=click.IntRange(min=1),
    show_default=True,
    default=DEF_CFG["sig_clu"]["thresh"],
    help="Minimum core size.",
)
@click.pass_context
def rsc(
    ctx,
    input_path,
    output_dir,
    input_type,
    res,
    markov_time,
    variable_markov_time,
    num_trials,
    seed,
    sig,
    cool_rate,
    thresh,
):
    """Community detection and significance clustering."""
    input_path = Path(input_path)
    input_type = InputData[input_type]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "bootstrap": {
            "seed": seed,
        },
        "sig_clu": {
            "cool_rate": cool_rate,
            "seed": seed,
            "sig": sig,
            "thresh": thresh,
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

        gui.subheader("Community detection")
        nops.partition(net)
        gui.info("Modules", nops.get_num_labels(net, "module"))
    elif input_path.is_dir():
        nets = [read_net(nops, path, input_type) for path in input_path.glob('*.csv')]

        gui.info("Replicate nets", len(nets))
        gui.report_average("Nodes", [len(bs_net.nodes) for bs_net in nets])
        gui.report_average("Edges", [len(bs_net.edges) for bs_net in nets])

        net = nets[0]

    gui.subheader("Network ensemble")
    if input_path.is_file():
        nets = nops.make_bootstrap_ensemble(net)
        gui.info("Resampled nets", len(nets))
        nodes = set(net.nodes)
    elif input_path.is_dir():
        nodes = set().union(*[set(n.nodes) for n in nets])

    parts = []
    for n in nets:
        nops.partition(n, set_node_attr=False)
        parts.append(nops.group_nodes_by_attr(n, "module"))

    gui.report_average("Modules", [len(p) for p in parts])

    gui.subheader("Significance clustering")
    cores = nops.sig_clu(nodes, parts)
    gui.info("Cores", len(cores))

    nops.associate_node_assignments(net, cores)
    nops.calc_node_centrality(net)

    # Save
    filename = f"{input_path.stem}_S{seed}"
    df = nops.to_dataframe(net)
    df.to_csv(output_dir / f"{filename}_nodes.csv", index=True)

    gplt = GeoPlot.from_dataframe(df)
    gplt.plot_structure()
    gplt.save(output_dir / f"{filename}_plt.png")

    usplt = UpSetPlot(cores, nets, parts, sig)
    usplt.save(output_dir / f"{filename}_upset.png")

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
@click.pass_context
def plot_structure(ctx, input_path, output_path):
    """Plots structure."""
    gplt = GeoPlot.from_file(input_path)
    gplt.plot_structure()

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