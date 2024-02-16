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

@dataclasses.dataclass
class GeoPlot:
    """Geospatial plotting."""
    gdf: gpd.GeoDataFrame
    fig: go.Figure = dataclasses.field(init=False)

    def plot(self, delineate_noise: bool = True) -> None:
        """Plots modular structure."""
        gdf = self.gdf
        self._color_modules(delineate_noise)
        gdf["node"] = gdf["node"].apply(hex)

        self.fig = go.Figure()
        geojson = json.loads(gdf.to_json())

        modules = gdf['module'].unique()
        for module in sorted(modules):
            for significance in [True, False] if delineate_noise else [True]:
                module_gdf = gdf[gdf["module"] == module]
                if delineate_noise:
                    module_gdf = module_gdf[module_gdf['significant'] == significance]

                color = module_gdf["color"].unique().item()
                # Add trace for significant or insignificant nodes
                self.fig.add_trace(go.Choropleth(
                    geojson=geojson,
                    locations=module_gdf.index,
                    z=module_gdf["module"],
                    name=module,
                    legendgroup=module,
                    showlegend=significance,
                    colorscale=[(0, color), (1, color)],
                    marker={"line": {"width": 0.5, "color": "black"}},
                    showscale=False,
                    customdata=module_gdf[["node"]],
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                    + "<extra></extra>"
                ))

        self._set_layout()
        self.fig.show()

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
            legend={
                "font_size": 12,
                "orientation": "h",
                "yanchor": "top",
                "y": 0.05,
                "xanchor": "right",
                "x": 0.98,
                "title_text": "Module",
                "itemsizing": "constant",
                "bgcolor": "rgba(255, 255, 255, 0)",
            },
        )

    def _color_modules(self, delineate_noise: bool = True, trivial_module_size: int = 1) -> None:
        """Assigns colors to modules based on significance, and marks trivial modules."""
        gdf = self.gdf
        gdf["module"] = gdf["module"].astype(str)

        modules_colors = {
            "1": {"significant": "#636EFA", "insignificant": "#A9B8FA"},
            "2": {"significant": "#EF553B", "insignificant": "#FAB9B5"},
            "3": {"significant": "#00CC96", "insignificant": "#80E2C1"},
            "4": {"significant": "#FFA15A", "insignificant": "#FFD1A9"},
            "5": {"significant": "#AB63FA", "insignificant": "#D4B5FA"},
            "6": {"significant": "#19D3F3", "insignificant": "#8CEAFF"},
            "7": {"significant": "#FF6692", "insignificant": "#FFB5C5"},
            "8": {"significant": "#B6E880", "insignificant": "#DAFAB6"},
            "9": {"significant": "#FF97FF", "insignificant": "#FFD1FF"},
            "10": {"significant": "#FECB52", "insignificant": "#FFE699"},
        }

        # Overwrite trivial module colors
        trivial_color = {"significant": "#A9A9A9", "insignificant": "#A9A9A9"}
        module_counts = gdf["module"].value_counts()
        trivial_modules = module_counts[module_counts <= trivial_module_size].index.tolist()
        for module in trivial_modules:
            modules_colors[module] = trivial_color

        if delineate_noise:
            if "significant" not in gdf.columns:
                raise ValueError(
                    "Node list must contain 'significant' column for significance coloring."
                )

            gdf["color"] = gdf.apply(
                lambda row: modules_colors[row["module"]]["significant"] if row["significant"]
                else modules_colors[row["module"]]["insignificant"],
                axis=1
            )
        else:
            # Default to significant colors
            gdf["color"] = gdf["module"].apply(
                lambda x: modules_colors[x]["significant"]
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
