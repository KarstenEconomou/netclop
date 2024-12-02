"""GeoPlot class."""
from json import loads
from os import PathLike
from typing import Optional, Self, Sequence

import geopandas as gpd
import h3.api.numpy_int as h3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shapely
from matplotlib.colors import LinearSegmentedColormap

from netclop.centrality import CentralityScale, centrality_registry
from netclop.constants import COLORS
from netclop.typing import NodeMetric, NodeSet, Partition


class GeoPlot:
    """Geospatial plotting."""
    def __init__(self, gdf: gpd.GeoDataFrame):
        self.gdf = gdf
        self.geojson = loads(self.gdf.to_json())

        self.fig: Optional[go.Figure] = None

    def save(self, path: Optional[PathLike]) -> None:
        """Save figure to static image."""
        if path is not None:
            # width = 6.75  # inches
            width = 3.375
            height = width / 2 # inches
            dpi = 900
            self.fig.write_image(path, width=width * dpi, height=height * dpi, scale=1.0, format="png")

    def show(self) -> None:
        """Show plot."""
        self.fig.show()

    def plot_structure(self, path: Optional[PathLike]=None) -> None:
        """Plot structure."""
        self.fig = go.Figure()

        self._color_node_core()
        for idx, trace_gdf in self._get_traces(self.gdf, "core"):
            self._add_trace_from_gdf(trace_gdf, str(idx))

        self._set_layout()
        self._set_legend()

        self.save(path)

    def plot_centrality(
        self,
        metric: NodeMetric,
        index: str="Centrality",
        path: Optional[PathLike]=None,
    ) -> None:
        """Plot centrality index."""
        self.fig = go.Figure()

        gdf = self.gdf
        gdf[index] = gdf["node"].map(metric)

        scale = centrality_registry.get(index).scale
        match scale:
            case CentralityScale.SEQUENTIAL:
                colorscale = px.colors.sequential.Viridis
                zmid = None
            case CentralityScale.DIVERGING:
                colorscale = px.colors.diverging.RdBu
                zmid = 0

        self.fig.add_trace(go.Choropleth(
            geojson=self.geojson,
            locations=gdf.index,
            z=gdf[index],
            zmid=zmid,
            marker={"line": {"width": 0.1, "color": "white"}},
            showscale=True,
            colorscale=colorscale,
            colorbar=dict(title=index.capitalize()),
        ))

        self._set_layout()

        self.save(path)

    def _get_traces(
        self,
        gdf: gpd.GeoDataFrame,
        col: str,
    ) -> list[tuple[str | int, gpd.GeoDataFrame]]:
        """Get all traces and corresponding labels to add to plot."""
        traces = []
        trace_idx = self._get_sorted_unique_col(gdf, col)
        for idx in trace_idx:
            trace_gdf = self._filter_to_col_entry(gdf, col, idx)
            traces.append((idx, trace_gdf))
        return traces

    def _add_trace_from_gdf(
        self,
        trace_gdf: gpd.GeoDataFrame,
        label: str,
        legend: bool=True,
    ) -> None:
        """Add trace to plot froma gpd.GeoDataFrame."""
        if not trace_gdf.empty:
            color = trace_gdf["color"].unique().item()

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
                "landcolor": "#EFEFDB",
                "showlakes": False,
                "showcountries": True,
            },
            hoverlabel={
                "bgcolor": "rgba(255, 255, 255, 0.8)",
                "font_size": 12,
                "font_family": "Arial",
            },
            margin={"r": 2, "t": 2, "l": 2, "b": 2},
        )

    def _set_legend(self) -> None:
        """Sets figure legend."""
        self.fig.update_layout(
            legend={
                "font_size": 24,
                "orientation": "h",
                "yanchor": "top",
                "y": 0.04,
                "xanchor": "right",
                "x": 0.96,
                "title_text": "",
                "itemsizing": "constant",
                "bgcolor": "rgba(255, 255, 255, 0)",
            },
        )

    def _color_node_core(self) -> None:
        """Assign a color to node corresponding to its core."""
        noise = "#CCCCCC"
        colors = {str(i): color for i, color in enumerate(COLORS, 1)}
        colors_fuzzy = {
            k: LinearSegmentedColormap.from_list("", [noise, color]) for k, color in colors.items()
        }

        n_colors = len(colors)
        self.gdf["color"] = self.gdf.apply(
            lambda node: colors[str((int(node["core"]) - 1) % n_colors + 1)] if node["core"]
            else noise,
            axis=1
        )

    @classmethod
    def from_cores(cls, cores: Partition, noise_nodes: Optional[NodeSet] = None) -> Self:
        """Make class instance from a set of cores."""
        core_nodes = [(node, i) for i, core in enumerate(cores, 1) for node in core]
        if noise_nodes is not None:
            core_nodes.extend([(node, 0) for node in noise_nodes])

        df = pd.DataFrame(core_nodes, columns=["node", "core"])
        return cls.from_dataframe(df)

    @classmethod
    def from_file(cls, path: PathLike) -> Self:
        """Make class instance from a file."""
        df = pd.read_csv(path)
        return cls.from_dataframe(df)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        """Make class instance from a pd.DataFrame."""
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
