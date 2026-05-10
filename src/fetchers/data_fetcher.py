import logging
import os

import geopandas as gpd

from src.config import DataPaths
from src.utils.json_io import write_json_atomic

WGS84 = "EPSG:4326"
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Handles loading and sampling of the Corine Land Cover (CLC) dataset scoped on Alsace

    """

    def __init__(
        self,
        data_path: str = DataPaths().corine_polygons,
    ):
        self.data_path = data_path
        self.gdf = None
        self._exclude_artificial = False

    def load_data(self, exclude_artificial: bool = False) -> gpd.GeoDataFrame:
        """Loads the dataset into memory"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        logger.info("Loading data from %s", self.data_path)
        raw_gdf = gpd.read_file(self.data_path)
        if "code_18" not in raw_gdf.columns:
            raise ValueError(f"Dataset at {self.data_path} must contain a code_18 column")
        if raw_gdf.crs is None:
            raw_gdf = raw_gdf.set_crs(WGS84)
        elif raw_gdf.crs != WGS84:
            raw_gdf = raw_gdf.to_crs(WGS84)
        raw_gdf["code_18"] = raw_gdf["code_18"].astype(str)
        self.gdf = (
            raw_gdf[~raw_gdf["code_18"].str.startswith("1")].copy()
            if exclude_artificial
            else raw_gdf
        )
        self._exclude_artificial = exclude_artificial
        return self.gdf

    def get_sample_polygons(
        self, n: int = 5, level: int = 2, exclude_artificial: bool = False
    ) -> gpd.GeoDataFrame:
        """
        Returns a sample of n polygons with their center point and their class.
        The level is the Corine Land Cover hierarchy level (1, 2, 3...)
        If exclude_artificial is True, polygons where code_18 starts with '1' are filtered out.

        """

        if self.gdf is None or self._exclude_artificial != exclude_artificial:
            self._exclude_artificial = exclude_artificial
            self.load_data(exclude_artificial=exclude_artificial)
        assert self.gdf is not None

        # we take a random sample
        sample = self.gdf.sample(n).copy()

        # then we extract the class code at the level desired. For this we use the code_18 column which contains the class label (e.g 311). To truncate this class label to the level desired we simply keep the number of digits wanted (e.g level 2 would keep the first two digits -> 31)
        sample["class_label"] = sample["code_18"].str[:level]

        # then we compute the polygon centroid
        sample["centroid"] = sample.to_crs(epsg=2154).centroid.to_crs(
            epsg=4326
        )  # we project to Lambert 93 (meters) to get an accurate centroid the we project back to WGS85 (degrees) for geocoding later.

        return sample[["class_label", "geometry", "centroid", "code_18"]]

    def get_bounds(self):
        """
        Return the bounding box containing all the polygons
        """

        if self.gdf is None:
            self.load_data()

        bounds = self.gdf.total_bounds
        return tuple(bounds)

    def save_bounds(self, output_path: str = DataPaths().corine_bounds):
        """
        Save the dataset bounding box to a json file
        """
        bounds = self.get_bounds()
        data = {
            "min_lon": bounds[0],
            "min_lat": bounds[1],
            "max_lon": bounds[2],
            "max_lat": bounds[3],
        }

        write_json_atomic(output_path, data, indent=4)
        logger.info("Bounds saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fetcher = DataFetcher()
    try:
        sample = fetcher.get_sample_polygons(n=3, exclude_artificial=True)
        logger.info("Succesfully sampled polygons:")
        logger.info("%s", sample[["class_label", "centroid", "code_18"]])
        fetcher.save_bounds()
    except Exception as e:
        logger.error("Error: %s", e)
