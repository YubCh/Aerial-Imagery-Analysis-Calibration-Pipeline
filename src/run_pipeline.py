from src.s1_datascan.scan import scan_dataset
from src.s2_statistics_data_mining.stats import build_stats_table, plot_distributions
from src.s2_statistics_data_mining.outlier import zscore,zscore_outliers, isolation_forest_outliers
from src.s5_storage.database import create_all_tables, stats_outliers_csv_to_database, ingest_embeddings, ingest_clusters


import pandas as pd

COLUMNS = ["brightness", "contrast", "blur_score"]


def pipeline():
  print("Running Pipeline")
  paths = scan_dataset()
  
  print("Computing Statistics")
  df = build_stats_table(paths)
  plot_distributions(df)
  df.to_csv("docs/image_stats.csv", index= False)

  print("Detecting Outliers")
  df =  zscore(df, COLUMNS)
  zs_if_outliers = pd.concat([zscore_outliers(df, COLUMNS), isolation_forest_outliers(df, COLUMNS)])
  zs_if_outliers.to_csv("docs/outliers.csv", index=False)


  print("ingest to Database")
  create_all_tables()
  stats_outliers_csv_to_database()
  ingest_embeddings()
  ingest_clusters()
  
if __name__ == "__main__":
  pipeline()