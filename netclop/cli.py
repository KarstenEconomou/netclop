import click

from netclop.networkops import NetworkOps
from netclop.plot import Plot

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
    default=5,
    show_default=True,
    required=False,
    help="H3 grid resolution (0-15) for domain discretization."
)
def construct_net(in_file: click.Path, out_file: click.Path, resolution: int) -> None:
    """Constructs a network from particle positions."""
    net = NetworkOps.from_locations(in_file, resolution)
    if out_file is not None:
        net.to_file(out_file)

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
    default=20,
    show_default=True,
    required=False,
    help="Number of outer trials to perform.",
)
@click.option("--markov-time", "-mt",
    type=float,
    default=1,
    show_default=True,
    required=False,
    help="Markov time spatial scale tuning parameter.",
)
@click.option("--seed", "-s",
    type=int,
    default=42,
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
    """Partitions a network."""
    net = NetworkOps.from_file(in_file)
    net.partition(num_trials, markov_time, seed)
    if out_file is not None:
        net.partition_to_file(out_file)

@netclop.command(name="plot")
@click.argument("in-file",
    type=click.Path(exists=True),
    required=True,
)
def plot(in_file: click.Path) -> None:
    """Plots a network partition."""
    plt = Plot.from_file(in_file)
    plt.plot()
