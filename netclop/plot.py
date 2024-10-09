"""Defines the GeoPlot and UpsetPlot class."""
import collections
import dataclasses
import itertools
import json
import typing

import geopandas as gpd
import h3.api.numpy_int as h3
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import shapely
from matplotlib.ticker import FormatStrFormatter
from upsetplot import UpSet

from .constants import Node, Partition, Path

DPI = 900

@dataclasses.dataclass
class GeoPlot:
    """Geospatial plotting."""
    gdf: gpd.GeoDataFrame
    fig: go.Figure = dataclasses.field(init=False)
    geojson: dict = dataclasses.field(init=False)

    def __post_init__(self):
        self._format_gdf()
        self.geojson = json.loads(self.gdf.to_json())
        self.fig = go.Figure()

    def save(self, path) -> None:
        """Saves figure to static image."""
        width = 5  # inches
        height = 3  # inches
        self.fig.write_image(path, height=height * DPI, width=width * DPI, scale=1)

    def show(self) -> None:
        """Shows plot."""
        self.fig.show()

    def plot_structure(self) -> None:
        """Plots structure."""
        gdf = self.gdf

        self._color_cores()
        for idx, trace_gdf in self._get_traces(gdf, "core"):
            self._add_trace(trace_gdf, str(idx))

        self._set_layout()
        self._set_legend()

    def plot_centrality(self) -> None:
        gdf = self.gdf
        centrality_indices = {
            "out_deg": "Out-degree",
            "in_deg": "In-degree",
            "out_str": "Out-strength",
            "in_str": "In-strength",
            "btwn": "Betweenness",
            "flow": "Flow",
        }

        customdata_columns = ["node", "module"] + list(centrality_indices.keys())
        customdata = gdf[customdata_columns]

        # Create a list of choropleth traces, one for each centrality index
        choropleth_traces = []
        for index, name in centrality_indices.items():
            hovertemplate_parts = [
                "<b>Node: </b>%{customdata[0]}",
                "<b>Module: </b>%{customdata[1]}<br>",
                "<b>Centrality</b>"
            ]

            for i, key in enumerate(centrality_indices.keys(), 2):
                if gdf[key].dtype == 'int':
                    format_str = f"{centrality_indices[key]}: %{{customdata[{i}]:,d}}"
                else:
                    format_str = f"{centrality_indices[key]}: %{{customdata[{i}]:.2e}}"
                hovertemplate_parts.append(format_str)

            hovertemplate = "<br>".join(hovertemplate_parts) + "<extra></extra>"

            choropleth_traces.append(go.Choropleth(
                geojson=self.geojson,
                locations=gdf.index,
                z=gdf[index],
                marker={"line": {"width": 0.1, "color": "white"}},
                showscale=True,
                colorbar=dict(title=name),
                colorscale="Viridis",
                customdata=customdata,
                hovertemplate=hovertemplate,
                visible=(index == list(centrality_indices.keys())[0]),
            ))

        for trace in choropleth_traces:
            self.fig.add_trace(trace)

        # Create buttons for the dropdown menu
        buttons = []
        for i, (index, name) in enumerate(centrality_indices.items()):
            buttons.append(dict(
                method="update",
                label=name,
                args=[
                    {"visible": [i == j for j in range(len(centrality_indices))]},
                    {"coloraxis": {"colorbar": {"title": name}}},
                ]
            ))

        self.fig.update_layout(
            updatemenus=[{
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
            }],
        )

        self._set_layout()

    def _format_gdf(self) -> None:
        """Formats gdf column types."""
        gdf = self.gdf
        gdf["module"] = gdf["module"].astype(str)
        # gdf["node"] = gdf["node"].astype(int).apply(hex)

    def _get_traces(
        self,
        gdf: gpd.GeoDataFrame,
        col: str,
    ) -> list[tuple[str | int, gpd.GeoDataFrame]]:
        """Operation to get all traces and corresponding labels to add to plot."""
        traces = []
        trace_idx = self._get_sorted_unique_col(gdf, col)
        for idx in trace_idx:
            trace_gdf = self._filter_to_col_entry(gdf, col, idx)
            traces.append((idx, trace_gdf))
        return traces

    def _add_trace(
        self,
        trace_gdf: gpd.GeoDataFrame,
        label: str,
        legend: bool=True,
        mute_trivial: bool=False,
    ) -> None:
        """Adds trace to plot."""
        if not trace_gdf.empty:
            color = trace_gdf["color"].unique().item()
            if legend and mute_trivial and len(trace_gdf) == 1:
                legend = False

            if label == "0":
                label = "Noise"

            self.fig.add_trace(go.Choropleth(
                geojson=self.geojson,
                locations=trace_gdf.index,
                z=trace_gdf["module"],
                name=label,
                legendgroup=label,
                showlegend=legend,
                colorscale=[(0, color), (1, color)],
                marker={"line": {"width": 0.1, "color": "white"}},
                showscale=False,
                customdata=trace_gdf[["node"]],
                hovertemplate="<b>%{customdata[0]}</b><br>"
                + "<extra></extra>"
            ))

    def _set_layout(self) -> None:
        """Sets basic figure layout with geography."""
        self.fig.update_layout(
            geo={
                "fitbounds": "locations",
                "projection_type": "natural earth",
                "resolution": 50,
                "showcoastlines": True,
                "coastlinecolor": "black",
                "coastlinewidth": 0.5,
                "showland": True,
                "landcolor": "#DCDCDC",
                "showlakes": False,
                "showcountries": True,
            },
            margin={"r": 2, "t": 2, "l": 2, "b": 2},
            hoverlabel={
                "bgcolor": "rgba(255, 255, 255, 0.8)",
                "font_size": 12,
                "font_family": "Arial",
            },
        )

    def _set_legend(self) -> None:
        """Sets figure legend."""
        self.fig.update_layout(
            legend={
                "font_size": 10,
                "orientation": "h",
                "yanchor": "top",
                "y": 0.05,
                "xanchor": "right",
                "x": 0.98,
                "title_text": "Core",
                "itemsizing": "constant",
                "bgcolor": "rgba(255, 255, 255, 0)",
            },
        )

    def _color_cores(self) -> None:
        """Assigns colors to cores."""
        gdf = self.gdf
        gdf["module"] = gdf["module"].astype(str)

        noise_color = "#CCCCCC"
        colors = {  # Core index zero reserved for noise
            "1": "#636EFA",
            "2": "#EF553B",
            "3": "#00CC96",
            "4": "#FFA15A",
            "5": "#AB63FA",
            "6": "#19D3F3",
            "7": "#FF6692",
            "8": "#B6E880",
            "9": "#FF97FF",
            "10": "#FECB52",
        }

        n_colors = len(colors)
        gdf["color"] = gdf.apply(
            lambda node: colors[str((int(node["core"]) - 1) % n_colors + 1)]
            if node["core"]
            else noise_color,
            axis=1
        )

        self.gdf = gdf

    @classmethod
    def from_file(cls, path: Path) -> typing.Self:
        """Make GeoDataFrame from file."""
        df = pd.read_csv(path)
        return cls.from_dataframe(df)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> typing.Self:
        """Make GeoDataFrame from DataFrame"""
        gdf = gpd.GeoDataFrame(df, geometry=cls._geo_from_cells(df["node"].values))
        return cls(gdf)

    @staticmethod
    def _geo_from_cells(cells: typing.Sequence[str]) -> list[shapely.Polygon]:
        """Get GeoJSON geometries from H3 cells."""
        return [
            shapely.Polygon(
                h3.cell_to_boundary(int(cell), geo_json=True)[::-1]
            ) for cell in cells
        ]

    @staticmethod
    def _reindex_modules(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Re-index module IDs ascending from South to North."""
        # Find the southernmost point for each module
        south_points = gdf.groupby("module")["geometry"].apply(
            lambda polygons: min(polygons, key=lambda polygon: polygon.bounds[1])
        ).apply(lambda polygon: polygon.bounds[1])

        # Sort the modules based on their southernmost points" latitude, in ascending order
        sorted_modules = south_points.sort_values(ascending=True).index

        # Re-index modules based on the sorted order
        module_id_mapping = {
            module: index - 1 for index, module in enumerate(sorted_modules, start=1)
        }
        gdf["module"] = gdf["module"].map(module_id_mapping)

        # Sort DataFrame
        gdf = gdf.sort_values(by=["module"], ascending=[True]).reset_index(drop=True)
        gdf["module"] = gdf["module"].astype(str)
        return gdf

    @staticmethod
    def _get_sorted_unique_col(gdf: gpd.GeoDataFrame, col: str) -> list:
        """Get all unique entries of a gdf column sorted."""
        return sorted(gdf[col].unique(), key=int)

    @staticmethod
    def _filter_to_col_entry(gdf: gpd.GeoDataFrame, col: str, entry) -> gpd.GeoDataFrame:
        """Get subset of gdf with column equal to a certain entry."""
        return gdf[gdf[col] == entry]

@dataclasses.dataclass
class UpSetPlot:
    cores: list[set[Node]]
    nets: list[nx.DiGraph]
    parts: list[Partition]
    sig: float

    def count_coalescence(self) -> dict[tuple[int, ...], int]:
        """Counts coalescence of cores across partitions."""
        counts = collections.defaultdict(int)

        for net, part in zip(self.nets, self.parts):
            prev_supcores = []
            for r in range(len(self.cores), 0, -1):
                for comb in itertools.combinations(enumerate(self.cores), r):
                    indices, sets = zip(*comb)

                    supcore = frozenset().union(*sets)  # Flatten cores to super-core
                    supcore_key = frozenset(indices)  # Key to identify super-core

                    # Check if mutually assigned
                    if any(supcore.issubset(module) for module in part):
                        # Check if comb is subset of a larger combination already counted
                        if not any(supcore_key.issubset(prev) for prev in prev_supcores):
                            prev_supcores.append(supcore_key)  # Save assignment
                            counts[supcore_key] += 1
        return counts

    def prep_data(self, counts: dict[tuple[int, ...], int]) -> pd.DataFrame:
        """Generates multi-index series from coalescence count data."""
        bools = list(itertools.product([True, False], repeat=len(self.cores)))
        labels = list(range(len(self.cores)))

        multi_index = pd.MultiIndex.from_tuples(bools, names=labels)
        data = pd.Series(0.0, index=multi_index)
        coherence = pd.Series([[] for _ in range(len(data))], index=multi_index)

        for key, count in counts.items():
            condition = pd.Series([True] * len(data), index=data.index)

            for label in labels:
                if label in key:
                    condition &= data.index.get_level_values(label)
                else:
                    condition &= ~data.index.get_level_values(label)
            data[condition] = count / len(self.parts)

        df = pd.DataFrame({'count': data})

        def sort_key(index_tuple):
            true_count = sum(index_tuple)
            order = [i for i, val in enumerate(index_tuple) if val]
            return true_count, order

        sorted_index = sorted(df.index, key=sort_key)
        df = df.reindex(sorted_index)

        df.index = pd.MultiIndex.from_tuples(df.index, names=labels)

        return df

    def plot(self, data: pd.DataFrame, path: Path) -> None:
        """Make UpSet plot."""
        plt.rc("font", family="Arial", size=10)
        upset = UpSet(
            data,
            sum_over="count",
            min_subset_size=self.sig,
            sort_by="cardinality",
            sort_categories_by="input",
            facecolor="black",
            shading_color=0.0,
            intersection_plot_elements=5,
            totals_plot_elements=2,
        )

        # Color shading by core
        colors = ["#636EFA", "#EF553B", "#00CC96", "#FFA15A", "#AB63FA",
                  "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]
        colors = [color.lstrip("#") for color in colors]
        colors = [tuple(int(color[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (0.5,) for color in colors]
        for label, color in zip(data.index.names, colors):
            upset.style_categories([label], shading_facecolor=color)

        # Setup fig and ax
        fig = plt.figure(figsize=(3.375, 3.375), dpi=900)
        ax = upset.plot(fig=fig)

        grid_linewidth = 0.25
        tick_linewidth = 0.5

        # Intersections
        ax["intersections"].set_ylabel("Coalescence frequency")
        ax["intersections"].axhline(y=(1-self.sig), color="gray", linestyle='--', linewidth=grid_linewidth)
        ax["intersections"].axhline(y=self.sig, color="gray", linestyle='--', linewidth=grid_linewidth)
        ax["intersections"].set_ylim(0.0, 1.0)
        ax["intersections"].grid(linewidth=grid_linewidth)
        ax["intersections"].yaxis.set_tick_params(width=tick_linewidth)
        ax["intersections"].spines["left"].set_linewidth(tick_linewidth)

        # Relabel cores from 1
        current_labels = [int(label.get_text()) for label in ax["matrix"].get_yticklabels()]
        new_labels = [str(label + 1) for label in current_labels]
        ax["matrix"].set_yticklabels(new_labels)

        # Totals
        ax["totals"].set_xlabel("Stability")
        ax["totals"].set_xlim(1.00, 1 - self.sig)
        ax["totals"].xaxis.set_tick_params(width=tick_linewidth)
        ax["totals"].spines["bottom"].set_linewidth(tick_linewidth)
        ax["totals"].grid(linewidth=grid_linewidth)

        plt.savefig(path, bbox_inches="tight")

    def save(self, path: Path):
        """Produce plot."""
        counts = self.count_coalescence()
        data = self.prep_data(counts)
        self.plot(data, path)

    @staticmethod
    def calculate_coherence(net: nx.DiGraph, nodes: set | frozenset) -> float:
        """Calculates the coherence ratio of a subset of nodes."""
        int_wgt, ext_wgt = 0, 0

        for src in nodes:
            for _, tgt, wgt in net.out_edges(src, data="weight"):
                if tgt in nodes:
                    int_wgt += wgt
                else:
                    ext_wgt += wgt

        if int_wgt == 0:
            return 0.0
        return int_wgt / (int_wgt + ext_wgt)
