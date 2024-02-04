"""Defines the GeoPlot class."""
import dataclasses
import json
import typing

import geopandas as gpd
import h3.api.numpy_int as h3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shapely

from .constants import Partition, Path

@dataclasses.dataclass
class GeoPlot:
    """Geospatial plotting."""
    gdf: gpd.GeoDataFrame
    fig: go.Figure = dataclasses.field(init=False)

    def plot(self) -> None:
        """Plot modular structure."""
        gdf = self.gdf

        modular_color_map = self._color_modules()
        self.fig = px.choropleth(
            gdf,
            geojson=gdf.geometry,
            locations=gdf.index,
            color="label",
            fitbounds="locations",
            projection="miller",
            custom_data=["node", "module"],
            color_discrete_map=modular_color_map,
        )
        self.fig.update_traces(
            marker={"opacity": 1.},
            hovertemplate=
            "<b>Index</b><br>"
            + "Cell %{customdata[0]}<br>"
            + "Module %{customdata[1]}<br>"
            + "<extra></extra>",
        )

        self._set_geos()
        self._set_layout()
        self.fig.show()

    def _set_layout(self) -> None:
        self.fig.update_layout(
            margin={"r": 0,"t": 0,"l": 0,"b": 0},
            hoverlabel={
                "bgcolor": "rgba(255, 255, 255, 1)",
                "font_size": 14,
                "font_family": "Arial"
            },
            legend=dict(
                yanchor="top",
                y=0.95,
                xanchor="left",
                x=0.01,
            ),
            legend_title="Module",
        )

    def _set_geos(self) -> None:
        self.fig.update_geos(
            resolution=50,
            showcoastlines=True,
            coastlinecolor="black",
            showland=True,
            landcolor="#deded1",
            showocean=True,
            oceancolor="white",
        )

    def _color_modules(self) -> dict[str, str]:
        """Assigns color to modules and noise."""
        gdf = self.gdf
        gdf["module"] = gdf["module"].astype(str)

        noise_label = "Noise"
        noise_color = "#e6e6e6"
        module_color_mapping = {
            "1": "#636EFA",
            "2": "#EF553B",
            "3": "#00CC96",
            "4": "#AB63FA",
            "5": "#FFA15A",
            "6": "#19D3F3",
            "7": "#FF6692",
            "8": "#B6E880",
            "9": "#FF97FF",
            "10": "#FECB52",
        }

        gdf["label"] = gdf.apply(
            lambda x: x["module"] if x["significant"] else noise_label,
            axis=1
        )

        # Sort labels
        sort_key = pd.Categorical(
        gdf['label'],
        categories=sorted(gdf['label'].unique(), key=lambda x: (x == noise_label, x)),
        ordered=True
        )

        self.gdf = gdf.assign(sort_key=sort_key).sort_values('sort_key').drop('sort_key', axis=1)
        return {**module_color_mapping, noise_label: noise_color}

    @classmethod
    def from_file(cls, path: Path) -> typing.Self:
        """Make GeoDataFrame from DataFrame."""
        df = pd.read_csv(path)
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
    def reindex_modules(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
