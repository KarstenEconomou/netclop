"""Commands for the CLI."""
import warnings
from pathlib import Path

import click

from netclop.constants import SEED
from netclop.ensemble.ensemble import NetworkEnsemble
from netclop.ensemble.sigclu import SigClu
from netclop.geo import GeoNet, GeoPlot
from netclop.ensemble.upsetplot import UpSetPlot

warnings.simplefilter(action="ignore", category=FutureWarning)


@click.command(name="rsc")
@click.argument(
    "paths",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    nargs=-1,
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help="Output directory.",
)
@click.option(
    "--seed",
    "-s",
    show_default=True,
    type=click.IntRange(min=1, max=None),
    default=SEED,
    help="Random seed.",
)
@click.option(
    "--res",
    type=click.IntRange(min=0, max=15),
    default=GeoNet.Config.res,
    show_default=True,
    help="H3 grid resolution for domain discretization.",
)
@click.option(
    "--markov-time",
    "-mt",
    type=click.FloatRange(min=0, max=None, min_open=True),
    default=NetworkEnsemble.Config.im_markov_time,
    show_default=True,
    help="Markov time to tune spatial scale of detected structure.",
)
@click.option(
    "--variable-markov-time/--static-markov-time",
    is_flag=True,
    show_default=True,
    default=NetworkEnsemble.Config.im_variable_markov_time,
    help="Permits the dynamic adjustment of Markov time with varying density.",
)
@click.option(
    "--num-trials",
    show_default=True,
    default=NetworkEnsemble.Config.im_num_trials,
    help="Number of outer-loop community detection trials to run.",
)
@click.option(
    "--sig",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    show_default=True,
    default=SigClu.Config.sig,
    help="Significance level for significance clustering.",
)
@click.option(
    "--cooling-rate",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    show_default=True,
    default=SigClu.Config.cooling_rate,
    help="Simulated annealing temperature cooling rate.",
)
@click.option(
    "--min-core-size",
    type=click.IntRange(min=1),
    show_default=True,
    default=SigClu.Config.min_core_size,
    help="Minimum core size.",
)
@click.option(
    "--plot-stability/--hide-stability",
    "plot_stability",
    is_flag=True,
    show_default=True,
    default=UpSetPlot.Config.plot_stability,
    help="Plots stability bars on the UpSet plot.",
)
@click.option(
    "--norm-counts/--abs-counts",
    "norm_counts",
    is_flag=True,
    show_default=True,
    default=UpSetPlot.Config.norm_counts,
    help="Shows normalized or absolute counts on the UpSet plot.",
)
@click.option(
    "--centrality",
    "-c",
    type=click.Choice(
        ["out-degree", "in-degree", "out-strength", "in-strength", "pagerank", "betweenness"],
        case_sensitive=False,
    ),
    multiple=True,
    help="Node centrality indices to compute and plot."
)
def rsc(
    paths,
    output_dir,
    res,
    markov_time,
    variable_markov_time,
    num_trials,
    seed,
    sig,
    cooling_rate,
    min_core_size,
    plot_stability,
    norm_counts,
    centrality,
):
    """Run recursive significance clustering from LPT positions."""
    # Make networks from LPT
    if len(paths) == 1:
        net = GeoNet(res=res).net_from_lpt(paths[0])
    else:
        gn = GeoNet(res=res)
        net = [gn.net_from_lpt(path) for path in paths]

    # Significance cluster network ensemble
    ne = NetworkEnsemble(
        net,
        seed=seed,
        im_markov_time=markov_time,
        im_variable_markov_time=variable_markov_time,
        im_num_trials=num_trials,
    )

    ne.sigclu(
        seed=seed,
        sig=sig,
        cooling_rate=cooling_rate,
        min_core_size=min_core_size,
        upset_config={
            "path": Path(output_dir) / "upset.png",
            "plot_stability": plot_stability,
            "norm_counts": norm_counts,
        },
    )

    # Plot structure
    gp = GeoPlot.from_cores(ne.cores, ne.unstable_nodes)
    gp.plot_structure(path=Path(output_dir) / "geo.png")

    # Plot centrality
    for centrality_index in centrality:
        gp.plot_centrality(
            ne.node_centrality(centrality_index),
            centrality_index,
            path=Path(output_dir) / f"centrality_{centrality_index.replace('-', '')}.png"
        )
