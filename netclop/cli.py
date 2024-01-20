import click

from netclop.networkops import NetworkOps
from netclop.plot import Plot

@click.command()
@click.argument("in-file",
    type=click.Path(exists=True),
    required=True
)
@click.option("--out-file", "-o",
    type=click.Path(),
    default=None,
    required=False,
    help="File to write network to"
)
@click.option("--resolution", "-res", "res",
    type=int,
    default=5,
    required=False,
    help="H3 grid resolution (0-15)"
)
def construct_net(in_file: click.Path, out_file: click.Path, res: int):
    net = NetworkOps.from_locations(in_file, res)
    if out_file is not None:
        net.to_file(out_file)

@click.command()
@click.argument("in-file",
    type=click.Path(exists=True),
    required=True
)
@click.option("--out-file", "-o",
    type=click.Path(),
    default=None,
    required=False,
    help="File to write partition to",
)
@click.option("--num-trials", "-n", "num_trials",
    default=20,
    type=int,
    required=False,
    help="Number of outer trials to perform",
)
@click.option("--markov-time", "-mt", "markov_time",
    default=1,
    type=float,
    required=False,
    help="Markov time spatial scale tuning parameter",
)
@click.option("--seed", "-s", "seed",
    default=42,
    type=int,
    required=False,
    help="Random seed",
)
def partition(in_file: click.Path, out_file: click.Path, num_trials: int, markov_time: float, seed: int):
    net = NetworkOps.from_file(in_file)
    net.partition(num_trials, markov_time, seed)
    if out_file is not None:
        net.partition_to_file(out_file)

@click.command()
@click.argument("in-file",
    type=click.Path(exists=True),
    required=True,
)
def plot(in_file: click.Path):
    plt = Plot.from_file(in_file)
    plt.plot()

@click.command()
@click.argument("in-file",
    type=click.Path(exists=True),
    required=True
)
@click.option("--resolution", "-res", "res",
    type=int,
    default=5,
    required=False,
    help="H3 grid resolution (0-15)"
)
@click.option("--num-trials", "-n", "num_trials",
    default=20,
    type=int,
    required=False,
    help="Number of outer trials to perform",
)
@click.option("--markov-time", "-mt", "markov_time",
    default=1,
    type=float,
    required=False,
    help="Markov time spatial scale tuning parameter",
)
@click.option("--seed", "-s", "seed",
    default=42,
    type=int,
    required=False,
    help="Random seed",
)
def quick_plot(in_file: click.Path, res: int, num_trials: int, markov_time: float, seed: int) -> None:
    net = NetworkOps.from_locations(in_file, res)
    net.partition(num_trials, markov_time, seed)
    plt = Plot.from_df(net.modular)
    plt.plot()
