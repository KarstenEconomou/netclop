"""Command line interface."""
import click

from .constants import DEFAULT_BIN_CONFIG, DEFAULT_PART_CONFIG, DEFAULT_BS_CONFIG, DEFAULT_SC_CONFIG
from .networkops import NetworkOps
from .plot import GeoPlot

@click.group()
def netclop():
    pass

@netclop.command(name="construct")
@click.argument("in-file",
    type=click.Path(exists=True),
    required=True
)
@click.option("--out-file", "-o",
    type=click.Path(),
    default=None,
    required=False,
    help="File to write network to."
)
@click.option("--resolution", "-res",
    type=int,
    default=DEFAULT_BIN_CONFIG["res"],
    show_default=True,
    required=False,
    help="H3 grid resolution (0-15) for domain discretization."
)
def construct_net(in_file: click.Path, out_file: click.Path, resolution: int) -> None:
    """Constructs a network from particle positions."""
    net = NetworkOps.from_locations(in_file, resolution)
    if out_file is not None:
        net.edge_list_to_file(out_file)

@netclop.command(name="partition")
@click.argument("in-file",
    type=click.Path(exists=True),
    required=True
)
@click.option("--out-file", "-o",
    type=click.Path(),
    default=None,
    required=False,
    help="File to write partition to.",
)
@click.option("--num-trials", "-n",
    type=int,
    default=DEFAULT_PART_CONFIG["num_trials"],
    show_default=True,
    required=False,
    help="Number of outer trials to perform.",
)
@click.option("--markov-time", "-mt",
    type=float,
    default=DEFAULT_PART_CONFIG["markov_time"],
    show_default=True,
    required=False,
    help="Markov time spatial scale tuning parameter.",
)
@click.option("--seed", "-s",
    type=int,
    default=DEFAULT_PART_CONFIG["seed"],
    show_default=True,
    required=False,
    help="Random seed.",
)
def partition(
    in_file: click.Path,
    out_file: click.Path,
    num_trials: int,
    markov_time: float,
    seed: int,
) -> None:
    """Clusters a network."""
    net = NetworkOps.from_file(
        in_file,
        part_config={
        "num_trials": num_trials,
        "markov_time": markov_time, 
        "seed": seed
        },
    )
    net.partition(path=out_file)

@netclop.command(name="sigclu")
@click.argument("in-file",
    type=click.Path(exists=True),
    required=True,
)
@click.option("--out-file", "-o",
    type=click.Path(),
    default=None,
    required=False,
    help="File to write node list to."
)
@click.option("--num-trials", "-n",
    type=int,
    default=DEFAULT_PART_CONFIG["num_trials"],
    show_default=True,
    required=False,
    help="Number of outer trials to perform.",
)
@click.option("--markov-time", "-mt",
    type=float,
    default=DEFAULT_PART_CONFIG["markov_time"],
    show_default=True,
    required=False,
    help="Markov time spatial scale tuning parameter.",
)
@click.option("--seed", "-s",
    type=int,
    default=DEFAULT_PART_CONFIG["seed"],
    show_default=True,
    required=False,
    help="Random seed.",
)
@click.option("--var-tune",
    type=float,
    default=DEFAULT_BS_CONFIG["tuning_param"],
    show_default=True,
    required=False,
    help="Variance tuning parameter for resampling.",
)
@click.option("--penalty-weight", "-pen",
    type=float,
    default=DEFAULT_SC_CONFIG["pen_weight"],
    show_default=True,
    required=False,
    help="Penalty weight in scoring solutions.",
)
@click.option("--cool-rate", "-cr",
    type=float,
    default=DEFAULT_SC_CONFIG["cool_rate"],
    show_default=True,
    required=False,
    help="Cooling rate for simulated annealing schedule.",
)
def sigclu(
    in_file: click.Path,
    out_file: click.Path,
    num_trials: int,
    markov_time: float,
    seed: int,
    var_tune: float,
    penalty_weight: float,
    cool_rate: float,
) -> None:
    """Finds the significant cores of network modular structure."""
    net = NetworkOps.from_file(
        in_file,
        part_config={
        "num_trials": num_trials,
        "markov_time": markov_time, 
        "seed": seed,
        },
        bs_config={
        "tuning_param": var_tune,
        "size": DEFAULT_BS_CONFIG["size"],
        "seed": seed,
        },
        sc_config={
        "conf": 0.05,
        "pen_weight": penalty_weight,
        "temp_init": DEFAULT_SC_CONFIG["temp_init"],
        "iter_max": DEFAULT_SC_CONFIG["iter_max"],
        "seed": seed,
        "cool_rate": cool_rate,
        },
    )
    net.significance_cluster(out_file)

@netclop.command(name="plot")
@click.argument("in-file",
    type=click.Path(exists=True),
    required=True,
)
def plot(in_file: click.Path) -> None:
    """Plots a network partition."""
    plt = GeoPlot.from_file(in_file)
    plt.plot()
