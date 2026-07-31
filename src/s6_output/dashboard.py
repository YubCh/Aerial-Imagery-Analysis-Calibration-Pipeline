import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
from src.s3_deep_learning_features.image_grouping import find_similar
from src.s4_calibration.image_correction import normalize_exposure
import cv2

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
with engine.connect() as conn:
  outlier_rows = conn.execute(text("""
    SELECT o.method, o.metric, o.score
    FROM outliers o
    JOIN images i ON i.id = o.image_id
    WHERE i.filename =:files
"""), {"files": selected}).fetchall()

  if outlier_rows:
    st.write("Flagged as outlier by:")
    for method, metric, score in outlier_rows:
      st.write(f" -{method} ({metric}), score {score:.2f}")
  else:
    st.write("Not flagged as outlier")
  st.write("Outlier example: 0000040_02454_d_0000068.jpg ")

st.header("Find similar images")

embeddings = np.load("docs/embeddings.npy")
emb_filenames = pd.read_csv("docs/embedding_filenames.csv")["filename"].tolist()

embs_index = emb_filenames.index(selected)
similar = find_similar(embs_index, embeddings,emb_filenames, 6)



st.write("Model resnet18 used with IMAGENET1K_V1 weights.")
st.write("Similar images clustered with kmeans algorithm.")
st.write("Most similar:")
cols = st.columns(3)
for i, (name, dist) in enumerate(similar[1:]):
  with cols[i % 3]:
    st.image(f"data/VisDrone2019-DET-train/images/{name}", width=200)
    st.write(f"distance: {dist:.2f}")

st.header("Callibration: normalize exposure")
img = cv2.imread(f"data/VisDrone2019-DET-train/images/{selected}")
corrected = normalize_exposure(img)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

col1, col2 = st.columns(2)
with col1:
  st.write("Original")
  st.image(img_rgb, width=300)

with col2:
  st.write("Exposure normalized")
  st.image(corrected_rgb, width=300)
