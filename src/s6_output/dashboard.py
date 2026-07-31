import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
from src.s3_deep_learning_features.image_grouping import find_similar

engine = create_engine("sqlite:///pipeline.db")

# images = Table(
#   "images", metadata,   
#   Column("id", Integer, primary_key=True),
#   Column("filename", Text, unique=True),
#   Column("width", Integer),
#   Column("height", Integer)
#   )

with engine.connect() as conn:
  filenames = []
  for row in conn.execute(text("""
    SELECT filename
    FROM images
    ORDER BY filename
""")):
    filenames.append(row[0])

# stats = Table("stats", metadata,
#   Column("image_id", Integer, ForeignKey("images.id")),
#   Column("brightness", Float),
#   Column("contrast", Float),
#   Column("blur_score", Float)
#   )
st.title("Aerial Imagery Analysis Pipeline")
st.write("Dashboard for dataset, statistics, and AI features.")

selected = st.selectbox("Select an image", filenames)
st.image(f"data/VisDrone2019-DET-train/images/{selected}")
with engine.connect() as conn:
  result = conn.execute(text("""
    SELECT s.brightness, s.contrast, s.blur_score
    FROM stats s
    JOIN images i ON i.id == s.image_id
    WHERE i.filename = :files"""), {"files": selected}).fetchone()



st.write("Brightness:", round(result[0], 2))
st.write("Contrast:", round(result[1], 2))
st.write("Blur-score:", round(result[2], 2))



st.header("Find similar images")

embeddings = np.load("docs/embeddings.npy")
emb_filenames = pd.read_csv("docs/embedding_filenames.csv")["filename"].tolist()

embs_index = emb_filenames.index(selected)
similar = find_similar(embs_index, embeddings,emb_filenames, 5)

st.write("Most similar:")
for name, dist in similar[1:]:
  st.image(f"data/VisDrone2019-DET-train/images/{name}", width=200)
  st.write(f"distance: {dist:.2f}")

