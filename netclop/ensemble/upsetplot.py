"""UpSetPlot class."""
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations, product
from os import PathLike

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from upsetplot import UpSet

from netclop.constants import COLORS
from netclop.typing import Partition


class UpSetPlot:
    """Class for constructing an UpSet plot."""
    @dataclass
    class Config:
        plot_stability: bool = True
        norm_counts: bool = True
        sig: float = 0.05
        opacity: float = 0.7

    def __init__(self, cores: Partition, partitions: list[Partition], **kwargs):
        self.cores = cores
        self.partitions = partitions
        self.cfg = self.Config(**kwargs)

        if self.max_stability == self.min_stability:
            self.cfg.plot_stability = False

    @cached_property
    def min_stability(self):
        """Minimum permissible stability of a core."""
        conf = 1 - self.cfg.sig
        min_stability = conf if self.cfg.norm_counts else np.ceil(len(self.partitions) * conf).astype(int)
        return min_stability

    @cached_property
    def max_stability(self):
        """Maximum possibility stability of a core."""
        return 1.0 if self.cfg.norm_counts else len(self.partitions)

    def _calc_coalescence_count(self) -> dict[tuple[int, ...], int]:
        """Counts coalescence of cores across partitions."""
        counts = defaultdict(int)

        for part in self.partitions:
            prev_supcores = []
            for r in range(len(self.cores), 0, -1):
                for comb in combinations(enumerate(self.cores), r):
                    indices, sets = zip(*comb)

                    supcore = frozenset().union(*sets)  # Flatten cores to super-core
                    supcore_key = frozenset(indices)  # Key to identify super-core

                    # Check if jointly assigned
                    if any(supcore <= module for module in part):
                        # Check if comb is subset of a larger combination already counted
                        if not any(supcore_key <= prev for prev in prev_supcores):
                            prev_supcores.append(supcore_key)  # Save assignment
                            counts[supcore_key] += 1
        return counts

    def _prep_data(self, counts: dict[tuple[int, ...], int]) -> pd.DataFrame:
        """Generates multi-index series from coalescence count data."""
        bools = list(product([True, False], repeat=len(self.cores)))
        labels = list(range(len(self.cores)))

        multi_index = pd.MultiIndex.from_tuples(bools, names=labels)
        data = pd.Series(0.0, index=multi_index)

        for key, count in counts.items():
            condition = pd.Series([True] * len(data), index=data.index)
            for label in labels:
                if label in key:
                    condition &= data.index.get_level_values(label)
                else:
                    condition &= ~data.index.get_level_values(label)
            data[condition] = count / len(self.partitions) if self.cfg.norm_counts else count

        df = pd.DataFrame({'count': data})

        def sort_key(index_tuple):
            true_count = sum(index_tuple)
            order = [i for i, val in enumerate(index_tuple) if val]
            return true_count, order

        sorted_index = sorted(df.index, key=sort_key)
        df = df.reindex(sorted_index)

        df.index = pd.MultiIndex.from_tuples(df.index, names=labels)

        return df

    def __color_cores(self, labels: list[str]) -> list[tuple[str, tuple[float, ...]]]:
        """Assign a color to each core."""
        n = len(labels)
        colors = (COLORS * (n // len(COLORS) + 1))[:n]  # Reuse colors if needed
        colors = [color.lstrip("#") for color in colors]
        colors = [tuple(int(color[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (self.cfg.opacity,) for color in colors]
        return [(label, color) for label, color in zip(labels, colors)]

    def _style_ax(self, ax: dict[str, plt.Axes], grid_lw: float = 0.25, tick_lw = 0.5) -> None:
        """Style UpSet plot axes,"""

        if self.cfg.norm_counts:
            ax["intersections"].set_ylabel("Coalescence frequency")
            ax["intersections"].set_ylim(0.0, 1.0)
        else:
            ax["intersections"].set_ylabel("Coalescence count")
            ax["intersections"].set_ylim(0, len(self.partitions))
        ax["intersections"].axhline(y=self.min_stability, color="gray", linestyle='--', linewidth=grid_lw)
        ax["intersections"].grid(linewidth=grid_lw)
        ax["intersections"].yaxis.set_tick_params(width=tick_lw)
        ax["intersections"].spines["left"].set_linewidth(tick_lw)

        current_labels = [int(label.get_text()) for label in ax["matrix"].get_yticklabels()]
        new_labels = [str(label + 1) for label in current_labels]
        ax["matrix"].set_yticklabels(new_labels)

        if self.cfg.plot_stability:
            #delta = (self.max_stability - self.min_stability) / 6
            delta = 0
            ax["totals"].set_xlabel("Stability")
            ax["totals"].set_xlim(self.max_stability, self.min_stability - delta)
            ax["totals"].set_xticks([self.max_stability, self.min_stability])
            ax["totals"].xaxis.set_tick_params(width=tick_lw)
            ax["totals"].spines["bottom"].set_linewidth(tick_lw)
            ax["totals"].grid(linewidth=grid_lw)

    def _get_min_coalescence_display(self):
        """Get the minimum coalescence frequency that should be displayed."""
        min_cf = self.max_stability - self.min_stability
        if min_cf == 0: min_cf = 1
        return min_cf

    def _plot(self, data: pd.DataFrame, path: PathLike) -> None:
        """Make UpSet plot."""
        plt.rc("font", family="Arial", size=10)
        upset = UpSet(
            data,
            sum_over="count",
            min_subset_size=self._get_min_coalescence_display(),
            sort_by="cardinality",
            sort_categories_by="input",
            facecolor="black",
            shading_color=0.0,
            intersection_plot_elements=5,
            totals_plot_elements=2 if self.cfg.plot_stability else 0,
        )

        for label, color in self.__color_cores(data.index.names):
            upset.style_categories([label], shading_facecolor=color)

        fig = plt.figure(figsize=(3.375, 3.375), dpi=900)
        ax = upset.plot(fig=fig)
        self._style_ax(ax)

        plt.savefig(path, bbox_inches="tight", format="png")

    def plot(self, path: PathLike):
        """Produce plot."""
        counts = self._calc_coalescence_count()
        data = self._prep_data(counts)
        self._plot(data, path)