from os import PathLike
from typing import Iterable, Sequence

import numpy as np
from tqdm.auto import tqdm
from loguru import logger

fmt = "<c>{time:YYYY-MM-DD HH:mm:ss.SSS}</c> | "\
      "<level>{level: <8}</level> | "\
      "<level>{message}</level>"

logger.remove()
logger.add(
    lambda msg: tqdm.write(msg, end=""),
    colorize=True,
    level="INFO",
    format=fmt,
)


class Logger:
    """Class for algorithm logging."""
    ascii = " =#"
    color = "WHITE"

    def __init__(self, path: PathLike = None, silent: bool = False):
        if path is not None:
            logger.add(path, colorize=False, format=fmt)
        self.silent = silent

    def log(self, msg: str, level="INFO", **kwargs) -> None:
        """Log info."""
        if not self.silent:
            match level:
                case "INFO": logger.opt(ansi=True).info(msg, **kwargs)
                case "DEBUG": logger.debug(msg, **kwargs)

    def pbar(self, iterable: Iterable, length: bool = True, **kwargs) -> tqdm | Iterable:
        """Make a tqdm progress bar."""
        if self.silent:
            return iterable

        if not length:
            iterable = iter(iterable)

        return tqdm(iterable, ascii=self.ascii, colour=self.color, **kwargs)

    # Manual progress bar
    def pbar_info(self, pbar: tqdm | Iterable, info: str) -> None:
        """Update information in progress bar."""
        if isinstance(pbar, tqdm):
            pbar.set_postfix_str(info)

    def make_pbar(self, **kwargs):
        """Make a tqdm progress bar for manual usage."""
        if not self.silent:
            return tqdm(ascii=self.ascii, colour=self.color, **kwargs)
        else:
            return None

    def update_pbar(self, pbar: tqdm, inc: int = 1, **kwargs):
        """Update manual instance of progress bar."""
        if isinstance(pbar, tqdm):
            pbar.update(inc)

    def close_pbar(self, pbar: tqdm | Iterable):
        """Close manual instance of progress bar."""
        if isinstance(pbar, tqdm):
            pbar.close()

    # Miscellaneous reporting utilities
    @staticmethod
    def stat(nums: Sequence) -> str:
        if len(nums) > 1:
            return f"{np.mean(nums):.1f}±{np.std(nums):.1f}"
        else:
            return nums[0]
