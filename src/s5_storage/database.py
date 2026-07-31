from sqlalchemy import Table, Column, create_engine, Table, Integer, Float, Text, ForeignKey, MetaData, text
import pandas as pd
from sqlalchemy import select
import os
if os.path.exists("pipeline.db"):
    os.remove("pipeline.db")
engine = create_engine("sqlite:///pipeline.db")
metadata = MetaData()

images = Table(
  "images", metadata,   
  Column("id", Integer, primary_key=True),
  Column("filename", Text, unique=True),
  Column("width", Integer),
  Column("height", Integer)
  )

stats = Table("stats", metadata,
  Column("image_id", Integer, ForeignKey("images.id")),
  Column("brightness", Float),
  Column("contrast", Float),
  Column("blur_score", Float)
  )

outliers = Table("outliers", metadata,
  Column("image_id", Integer, ForeignKey("images.id")),
  Column("method", Text),
  Column("metric", Text),
  Column("score", Float)
  )

embeddings = Table("embeddings", metadata,
  Column("image_id", Integer, ForeignKey("images.id")),
  Column("npy_path", Text),
  Column("vector_index", Integer)
  )


clusters = Table("clusters", metadata,
  Column("image_id", Integer, ForeignKey("images.id")),
  Column("cluster_id", Integer))

def create_all_tables():
  metadata.create_all(engine)
  print("tables created in pipeline.db")

def ingest_clusters():
  df_clust = pd.read_csv("docs/clusters.csv")
  with engine.connect() as conn:
    existing = conn.execute(select(images.c.id, images.c.filename))
    filename_to_id = {}
    for img_id, filename in existing:
      filename_to_id[filename] = img_id
    for _, row in df_clust:
      image_id = filename_to_id[row["filenames"]]
      cluster = row["cluster"]
      conn.execute(clusters.insert().values(
        image_id = image_id,
        cluster_id = cluster
      ))

def ingest_embeddings():
  df_emb = pd.read_csv("docs/embedding_filenames.csv")
  with engine.begin() as connection:
    existing = connection.execute(select(images.c.id, images.c.filename))
    filename_to_id = {}
    for img_id, filename in existing:
      filename_to_id[filename] = img_id
    for i, row in df_emb.iterrows():
      image_id = filename_to_id[row["filename"]]
      connection.execute(embeddings.insert().values(
       image_id = image_id,
       npy_path = "docs/embeddings.npy",
       vector_index = i
      ))

def stats_outliers_csv_to_database(stats_csv_path="docs/image_stats.csv",      outliers_csv_path = "docs/outliers.csv"):
  df_stats = pd.read_csv(stats_csv_path)
  df_outliers = pd.read_csv(outliers_csv_path)
  filename_to_id = {}
  with engine.begin() as connection:
    for _, row in df_stats.iterrows():
      result = connection.execute(images.insert().values(
        filename = row["filename"],
        width = row["width"],
        height = row["height"]
      ))
      image_id = result.inserted_primary_key[0]
      filename_to_id[row["filename"]] = image_id
      connection.execute(stats.insert().values(
        image_id = image_id,
        brightness = row["brightness"],
        contrast = row["contrast"],
        blur_score = row["blur_score"]
      ))

    for _, row in df_outliers.iterrows():
      connection.execute(outliers.insert().values(
        image_id = filename_to_id[row["filename"]],
        method = row["method"],
        metric = row["metric"],
        score = row["score"]
      ))

      
  print(f"ingested {len(df_stats)} stats")
  print(f"ingested {len(df_outliers)} outliers")




def run_example_queries():
  with engine.connect() as connection:
    blurry_images = connection.execute(text("""
      SELECT i.filename, s.blur_score
      FROM stats s
      JOIN images i ON i.id = s.image_id
      ORDER BY s.blur_score
      LIMIT 10
  """))
    for row in blurry_images:
      print(row)

    outlier_method_count = connection.execute(text("""
      SELECT COUNT(o.method)
      FROM outliers o
      GROUP BY o.method
"""))
    for row in outlier_method_count:
      print(row)

    average_brightness = connection.execute(text("""
      SELECT AVG(s.brightness)
      FROM stats s
"""))
    for row in average_brightness:
      print(row)

    zscore_ironforest_outlier = connection.execute(text("""
      SELECT DISTINCT i.filename
      FROM images i
      WHERE i.id IN (SELECT image_id FROM outliers WHERE method = 'zscore')
        AND i.id IN (SELECT image_id FROM outliers WHERE method = 'isolation_forest')
"""))
    for row in zscore_ironforest_outlier:
      print(row)


if __name__ == "__main__":
    run_example_queries()