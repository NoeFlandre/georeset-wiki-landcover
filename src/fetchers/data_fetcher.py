import json
import os

import geopandas as gpd


class DataFetcher:
    """
    Handles loading and sampling of the Corine Land Cover (CLC) dataset scoped on Alsace

    """

    def __init__(
        self, data_path: str = "data/corine/alsace_corine_land_use_2018/occupation_sol_2018.shp"
    ):
        self.data_path = data_path
        self.gdf = None
        self._exclude_artificial = False

    def load_data(self, exclude_artificial: bool = False) -> gpd.GeoDataFrame:
        """Loads the dataset into memory"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        print(f"Loading data from {self.data_path} ...")
        raw_gdf = gpd.read_file(self.data_path)
        self.gdf = raw_gdf[~raw_gdf["code_18"].str.startswith("1")].copy() if exclude_artificial else raw_gdf
        self._exclude_artificial = exclude_artificial
        return self.gdf

    def get_sample_polygons(self, n: int = 5, level: int = 2, exclude_artificial: bool = False) -> gpd.GeoDataFrame:
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

    def save_bounds(self, output_path: str = "data/corine/bounds.json"):
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

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Bounds saved to {output_path}")


if __name__ == "__main__":
    fetcher = DataFetcher()
    try:
        sample = fetcher.get_sample_polygons(n=3, exclude_artificial=True)
        print("Succesfully sampled polygons:")
        print(sample[["class_label", "centroid", "code_18"]])
        fetcher.save_bounds()
    except Exception as e:
        print(f"Error: {e}")
