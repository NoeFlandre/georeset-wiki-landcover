""" Summarize OSM polygons class distribution counts """

import pandas as pd 

def class_count_summary(csv_path: str) -> pd.DataFrame:

    """
    For each OSM polygon, we count how many CORINE classes intersect it.
    We return a summary table of how many polygons have n classes.
    """

    df = pd.read_csv(csv_path)

    class_counts = df.groupby("osm_id").size()

    summary = (
        class_counts
        .value_counts()
        .sort_index()
        .reset_index()
    )

    summary.columns = ["n_classes", "n_polygons"]
    summary["pct_polygons"] = (summary["n_polygons"] / len(class_counts)*100).round(2)
    return summary

if __name__=="__main__":
    summary = class_count_summary("data/distribution/osm_corine_distribution.csv")
    print(summary.to_string(index=False))