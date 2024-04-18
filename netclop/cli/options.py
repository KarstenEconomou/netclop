"""Arguments and options for the CLI."""
import functools

import click

from ..config_loader import load_config
from ..sigcore import SigCluScheme

DEF_CFG = load_config()

def io(f):
    """Input and output argument and option."""
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
    @functools.wraps(f)
    def wrapper_path_options(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper_path_options


def binning(f):
    """Binning options."""
    @click.option(
        "--res",
        type=click.IntRange(min=0, max=15),
        default=DEF_CFG["binning"]["res"],
        show_default=True,
        help="H3 grid resolution for domain discretization.",
    )
    @functools.wraps(f)
    def wrapper_path_options(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper_path_options


def comm_detection(f):
    """Community detection options."""
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
    @functools.wraps(f)
    def wrapper_path_options(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper_path_options

def sc_scheme(f):
    """Scheme options."""
    @click.option(
        "--significance-cluster", 
        "-sc",
        "sc_scheme",
        type=click.Choice([scheme.name for scheme in SigCluScheme], case_sensitive=False),
        default="NONE",
        show_default=True,
        help="Scheme to demarcate significant community assignments from statistical noise.",
    )
    @functools.wraps(f)
    def wrapper_path_options(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper_path_options


def sig_clu(f):
    """Significance clustering scheme and options."""
    @sc_scheme
    @click.option(
        "--cooling-rate",
        "-cr",
        "cool_rate",
        type=float,
        show_default=True,
        default=DEF_CFG["sig_clu"]["cool_rate"],
        help="Cooling rate in simulated annealing.",
    )
    @functools.wraps(f)
    def wrapper_path_options(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper_path_options


def plot(f):
    """Plotting options."""
    @functools.wraps(f)
    def wrapper_path_options(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper_path_options
