import click

from tranetclu.networkops import NetworkOps

@click.command()
@click.argument("file", type=str, required=True)
@click.option("--outfile", ype=str, required=True, help="File to write network to")
@click.option("--resolution", "-res", default=5, type=int, required=False, help="H3 grid resolution (0-15)")
def make(in_file: str, out_file: str, res: int):
    click.echo("Hello makenet!")
    net = NetworkOps.from_locations(in_file, res)
    net.to_file(out_file)

@click.command()
@click.argument("file", type=str, required=True)
@click.option("--outfile", ype=str, required=True, help="File to write partition to")
@click.option("--num_trials", default=20, type=int, required=False, help="Number of outer trials to perform")
@click.option("--markov_time", default=1, type=float, required=False, help="Markov time tuning parameter")
@click.option("--seed", default=42, type=int, required=False, help="Random seed")
def partition(file: str, num_trials: int, markov_time: float, seed: int):
    net = NetworkOps.from_file(file)
    net.partition(num_trials, markov_time, seed)