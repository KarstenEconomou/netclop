import click
import dataclasses
import typing

import geopandas as gpd
import h3.api.numpy_int as h3
import pandas as pd
import plotly.express as px
import shapely

@dataclasses.dataclass
class Plot:
    """Geospatial plotting."""
    gdf: gpd.GeoDataFrame

    def plot(self) -> None:
        """Plot modular structure."""
        gdf = self.gdf

        def make_color_map(modules: list[str]) -> dict[str, str]:
            color_seq = px.colors.qualitative.Plotly
            return dict((mod, col) for mod,col in zip(modules, color_seq))

        color_map = make_color_map(gdf["module"].unique().tolist())

        fig = px.choropleth(
            gdf,
            geojson=gdf.geometry,
            locations=gdf.index,
            color="module",
            fitbounds="locations",
            projection="miller",
            custom_data=["node", "module", "flow"],
            color_discrete_map=color_map,
        )
        fig.update_traces(
            marker={"opacity": 0.93},
            hovertemplate=
            "<b>Index</b><br>"
            + "Cell %{customdata[0]}<br>"
            + "Module %{customdata[1]}<br>"
            + "<b>Centrality Indices</b><br>"
            + "PageRank: %{customdata[2]:.1e}<br>"
            + "<extra></extra>",
        )
        fig.update_geos(
            resolution=50,
            showcoastlines=True, 
            coastlinecolor="black",
            showland=True, 
            landcolor="#deded1",
            showocean=True, 
            oceancolor="white",
        )
        fig.update_layout(
            margin={"r": 0,"t": 0,"l": 0,"b": 0},
            hoverlabel={
                "bgcolor": "rgba(255, 255, 255, 1)",
                "font_size": 14,
                "font_family": "Arial"
            },
        )
        fig.show()

    @classmethod
    def from_file(cls, path: click.Path, cull_trivial: bool = False) -> typing.Self:
        """Make GeoDataFrame from file."""
        df = pd.read_csv(path, index_col=False, names=["node", "module", "flow"])
        return cls.from_df(df, cull_trivial=cull_trivial)
    
    @classmethod
    def from_df(cls, df: pd.DataFrame, cull_trivial: bool = False) -> typing.Self:
        """Make GeoDataFrame from DataFrame."""
        if cull_trivial:
            trivial_modules = df["module"].value_counts()[df["module"].value_counts() == 1].index
            df = df[~df["module"].isin(trivial_modules)]

        gdf = gpd.GeoDataFrame(df, geometry=cls.geometry_from_cells(df["node"].values))
        gdf = cls.reindex_modules(gdf)
        return cls(gdf)

    @staticmethod
    def geometry_from_cells(cells: typing.Sequence[str]) -> list[shapely.Polygon]:
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
        module_id_mapping = {module: index - 1 for index, module in enumerate(sorted_modules, start=1)}
        gdf["module"] = gdf["module"].map(module_id_mapping)

        # Sort DataFrame
        gdf = gdf.sort_values(by=["module", "flow"], ascending=[True, False]).reset_index(drop=True)
        gdf["module"] = gdf["module"].astype(str)
        return gdf
