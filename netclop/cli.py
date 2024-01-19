import click

from tranetclu.networkops import NetworkOps

@click.command()
@click.argument("in-file", type=click.Path(exists=True), required=True)
@click.option("--out-file", "-o", type=click.Path(), required=True, help="File to write network to")
@click.option("--resolution", "-res", "res", default=5, type=int, required=False, help="H3 grid resolution (0-15)")
def make_net(in_file: click.Path, out_file: click.Path, res: int):
    click.echo("Hello makenet!")
    net = NetworkOps.from_locations(in_file, res)
    net.to_file(out_file)

@click.command()
@click.argument("in-file", type=click.Path(exists=True), required=True)
@click.option("--out-file", "-o", type=click.Path(), required=True, help="File to write partition to")
@click.option("--num-trials", "-n", "num_trials", default=20, type=int, required=False, help="Number of outer trials to perform")
@click.option("--markov-time", "-mt", "markov_time", default=1, type=float, required=False, help="Markov time tuning parameter")
@click.option("--seed", "-s", "seed", default=42, type=int, required=False, help="Random seed")
def partition(in_file: click.Path, out_file: click.Path, num_trials: int, markov_time: float, seed: int):
    net = NetworkOps.from_file(in_file)
    net.partition(out_file, num_trials, markov_time, seed)
