"""Defines the GeoPlot and UpsetPlot class."""
from dataclasses import dataclass
from json import loads
from os import PathLike
from typing import Self, Sequence

import geopandas as gpd
import h3.api.numpy_int as h3
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import shapely

from .constants import Node, Partition, COLORS


class GeoNet:
    @dataclass(frozen=True)
    class Config:
        res: int = 5

    type Cell = int

    def __init__(self, **config_options):
        self.cfg = self.Config(**config_options)

    def bin_positions(self, lngs: Sequence[float], lats: Sequence[float]) -> list[Cell]:
        """Bin (lng, lat) coordinate pairs into an H3 cell."""
        return [h3.latlng_to_cell(lat, lng, self.cfg.res) for lat, lng in zip(lats, lngs)]

    def make_lpt_edges(self, path: PathLike) -> tuple[tuple[Cell, Cell], ...]:
        """Make an edge list (with duplicates) from LPT positions."""
        data = pd.read_csv(
            path,
            names=["initial_lng", "initial_lat", "final_lng", "final_lat"],
            index_col=False,
            comment="#",
        )

        srcs = self.bin_positions(data["initial_lng"], data["initial_lat"])
        tgts = self.bin_positions(data["final_lng"], data["final_lat"])
        return tuple(zip(srcs, tgts))

    def net_from_lpt(self, path: PathLike) -> nx.DiGraph:
        """Construct a network from LPT positions."""
        net = nx.DiGraph()
        edges = self.make_lpt_edges(path)

        for src, tgt in edges:
            if net.has_edge(src, tgt):
                # Record another transition along a recorded edge
                net[src][tgt]["weight"] += 1
            else:
                # Record a new edge
                net.add_edge(src, tgt, weight=1)

        nx.relabel_nodes(net, dict((name, str(name)) for name in net.nodes), copy=False)
        return net


class GeoPlot:
    """Geospatial plotting."""
    def __init__(self, gdf: gpd.GeoDataFrame):
        self.gdf = gdf
        self.geojson = loads(self.gdf.to_json())
        self.fig = go.Figure()

    def save(self, path: PathLike) -> None:
        """Saves figure to static image."""
        width = 5  # inches
        height = 3  # inches
        dpi = 900
        self.fig.write_image(path, height=height * dpi, width=width * dpi, scale=1)

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
                z=trace_gdf["core"],
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
        # gdf["module"] = gdf["module"].astype(str)

        noise_color = "#CCCCCC"
        colors = dict((str(i), color) for i, color in enumerate(COLORS, 1))

        n_colors = len(colors)
        gdf["color"] = gdf.apply(
            lambda node: colors[str((int(node["core"]) - 1) % n_colors + 1)]
            if node["core"]
            else noise_color,
            axis=1
        )

        self.gdf = gdf

    @classmethod
    def from_partition(cls, nodes: set[Node], partition: Partition):
        core_nodes = [(node, i) for i, core in enumerate(partition, 1) for node in core]
        core_nodes.extend([(node, 0) for node in nodes.difference(set().union(*partition))])

        df = pd.DataFrame(core_nodes, columns=["node", "core"])
        return cls.from_dataframe(df)

    @classmethod
    def from_file(cls, path: PathLike) -> Self:
        """Make class instance from a file."""
        df = pd.read_csv(path)
        return cls.from_dataframe(df)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        """Make class instance from a pd.DataFrame"""
        gdf = gpd.GeoDataFrame(df, geometry=cls._geo_from_cells(df["node"].values))
        return cls(gdf)

    @staticmethod
    def _geo_from_cells(cells: Sequence[str]) -> list[shapely.Polygon]:
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
