import numpy as np
from click import echo

HEAD_WIDTH: int = 68
INFO_WIDTH: int = 16


def header(title: str) -> None:
    """Makes header."""
    echo("=" * HEAD_WIDTH)
    echo(f"{title:^{HEAD_WIDTH}}")
    echo("=" * HEAD_WIDTH)


def subheader(title: str) -> None:
    """Makes subheader."""
    echo("")
    echo(f"{title:^{HEAD_WIDTH}}")
    echo("-" * HEAD_WIDTH)


def info(title: str, disp) -> None:
    """Prints line of info."""
    if type(disp) is int: disp = f"{disp:,}"

    echo(f" - {title:<{INFO_WIDTH}}: {disp}")


def footer() -> None:
    echo("")


def report_average(title: str, items: list) -> None:
    """Reports the average of items."""
    info(title, f"{np.mean(items):.1f} ± {np.std(items):.1f}")
