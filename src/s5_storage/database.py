from sqlalchemy import Table, Column, create_engine, Table, Integer, Float, Text, ForeignKey, MetaData

engine = create_engine("sqllite:///pipeline.db")
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


def create_all_tables():
  metadata.create_all(engine)
  print("tables created in pipeline.db")

