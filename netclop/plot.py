"""Defines the GeoPlot class."""
import dataclasses
import json
import typing

import geopandas as gpd
import h3.api.numpy_int as h3
import pandas as pd
import plotly.graph_objects as go
import shapely

from .constants import Path
from .sigcore import SigCluScheme

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
        dpi = 300
        width = 5  # inches
        height = 3  # inches
        self.fig.write_image(path, height=height * dpi, width=width * dpi, scale=1)

    def show(self) -> None:
        """Shows plot."""
        self.fig.show()

    def plot_structure(
        self,
        sc_scheme: SigCluScheme=SigCluScheme.NONE,
        mute_trivial: bool = False,
    ) -> None:
        """Plots structure."""
        gdf = self.gdf

        self._color_modules(sc_scheme, mute_trivial)

        match sc_scheme:
            case SigCluScheme.NONE:
                for idx, trace_gdf in self._get_traces(gdf, "module"):
                    self._add_trace(trace_gdf, idx)
                legend_title = "Module"
            case SigCluScheme.STANDARD:
                for module_idx, module_gdf in self._get_traces(gdf, "module"):
                    for core_idx, core_gdf in self._get_traces(module_gdf, "core"):
                        self._add_trace(core_gdf, module_idx, legend=bool(core_idx))
                legend_title = "Module"
            case SigCluScheme.RECURSIVE:
                for idx, trace_gdf in self._get_traces(gdf, "core"):
                    self._add_trace(trace_gdf, str(idx))
                legend_title = "Core"

        self._set_layout()
        self._set_legend(legend_title)

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

    def _set_legend(self, title: str) -> None:
        """Sets figure legend."""
        self.fig.update_layout(
            legend={
                "font_size": 12,
                "orientation": "h",
                "yanchor": "top",
                "y": 0.05,
                "xanchor": "right",
                "x": 0.98,
                "title_text": title,
                "itemsizing": "constant",
                "bgcolor": "rgba(255, 255, 255, 0)",
            },
        )

    def _color_modules(self, sig_clu: SigCluScheme, mute_trivial: bool = False) -> None:
        """Assigns colors to modules based on significance, and marks trivial modules."""
        gdf = self.gdf
        gdf["module"] = gdf["module"].astype(str)

        noise_color = "#CCCCCC"
        colors = {  # Core index zero reserved for recursive noise
            "1": {"core": "#636EFA", "noise": "#A9B8FA"},
            "2": {"core": "#EF553B", "noise": "#FAB9B5"},
            "3": {"core": "#00CC96", "noise": "#80E2C1"},
            "4": {"core": "#FFA15A", "noise": "#FFD1A9"},
            "5": {"core": "#AB63FA", "noise": "#D4B5FA"},
            "6": {"core": "#19D3F3", "noise": "#8CEAFF"},
            "7": {"core": "#FF6692", "noise": "#FFB5C5"},
            "8": {"core": "#B6E880", "noise": "#DAFAB6"},
            "9": {"core": "#FF97FF", "noise": "#FFD1FF"},
            "10": {"core": "#FECB52", "noise": "#FFE699"},
        }
        n = len(colors)

        def get_color(idx):
            return colors[str((int(idx) - 1) % n + 1)]

        match sig_clu:
            case SigCluScheme.NONE:
                module_counts = gdf["module"].value_counts()

                gdf["color"] = gdf.apply(
                    lambda node: get_color(node["module"])["core"]
                    if not (module_counts[node["module"]] == 1 and mute_trivial)
                    else noise_color,
                    axis=1
                )
            case SigCluScheme.STANDARD:
                gdf["color"] = gdf.apply(
                    lambda node: get_color(node["module"])["core"]
                    if node["core"]
                    else get_color(node["module"])["noise"],
                    axis=1
                )
            case SigCluScheme.RECURSIVE:
                gdf["color"] = gdf.apply(
                    lambda node: get_color(node["core"])["core"]
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
