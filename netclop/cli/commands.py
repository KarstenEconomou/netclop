"""Commands for the CLI."""
from importlib.metadata import version
import warnings
from pathlib import Path

import click

from netclop.centrality.centrality import centrality_registry
from netclop.constants import SEED
from netclop.ensemble.ensemble import NetworkEnsemble
from netclop.ensemble.sigclu import SigClu
from netclop.ensemble.upsetplot import UpSetPlot
from netclop.geo import GeoNet, GeoPlot
from netclop.log import Logger
from netclop.cli.files import make_run_id, make_filepath

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
    type=click.Choice(centrality_registry.registered, case_sensitive=False),
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
    """Run recursive significance clustering from LPT simulations."""
    # Set up run and logging
    run_id = make_run_id(seed, sig)
    path = Path(output_dir) / run_id
    logger = Logger(path=make_filepath(path, extension="log"))
    logger.log(f"<y>netclop v{version("netclop")}: run {run_id}</y>")
    logger.log(f"LPT paths {paths}", level="DEBUG")
    logger.log(f"output path '{output_dir}'", level="DEBUG")

    # Make networks from LPT
    net = GeoNet(res=res, logger=logger).from_lpt(paths)

    # Significance cluster network ensemble
    ne = NetworkEnsemble(
        net,
        seed=seed,
        im_markov_time=markov_time,
        im_variable_markov_time=variable_markov_time,
        im_num_trials=num_trials,
        logger=logger,
    )
    ne.sigclu(
        seed=seed,
        sig=sig,
        cooling_rate=cooling_rate,
        min_core_size=min_core_size,
        upset_config={
            "path": make_filepath(path, "upset"),
            "plot_stability": plot_stability,
            "norm_counts": norm_counts,
        },
    )

    # Plot structure
    logger.log("Plotting spatially-embedded cores.")
    gp = GeoPlot.from_cores(ne.cores, ne.unstable_nodes)
    gp.plot_structure(path=make_filepath(path, "geo"))

    # Plot centrality
    metrics = dict()
    if len(centrality) > 0:
        logger.log("Computing and plotting node centrality indices.")
        for index in logger.pbar(centrality):
            metrics[index] = ne.node_centrality(index)
            gp.plot_centrality(
                metrics[index],
                index,
                path=make_filepath(path, f"c_{index.replace('-', '')}")
            )

    logger.log("Saving node list.")
    ne.to_nodelist(metrics, path=make_filepath(path, extension="csv"))
