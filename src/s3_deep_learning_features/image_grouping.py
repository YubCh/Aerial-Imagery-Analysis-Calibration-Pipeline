import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import shutil
from pathlib import Path
from sklearn.cluster import DBSCAN



# clusterer = DBSCAN(eps=10, min_samples=5)
# cluster_ids = clusterer.fit_predict(embeddings) 
def cluster_images(embeddings, n_clusters=8):
  kmeans = KMeans(n_clusters=n_clusters, random_state=42)
  cluster_ids = kmeans.fit_predict(embeddings)
  return cluster_ids



def export_cluster_samples(cluster_ids, filenames, cluster_to_show, n=12):
  out_dir = Path(f"docs/cluster_{cluster_to_show}_samples")
  out_dir.mkdir(exist_ok=True)
  src_dir = Path("data/VisDrone2019-DET-train/images")
  count = 0
  for i, cid in enumerate(cluster_ids):
      if cid == cluster_to_show and count < n:
          shutil.copy(src_dir / filenames[i], out_dir / filenames[i])
          count += 1
  print(f"copied {count} cluster {cluster_to_show} to {out_dir}")



def find_similar(embs_index, embs, filenames, top_n=5):
  single_embedding = embs[embs_index]
  distances = np.linalg.norm(embeddings - single_embedding, axis=1)
  nearest = np.argsort(distances)[:top_n+1]
  res = []
  for i in nearest:
    res.append((filenames[i],distances[i]))
  return res




if __name__ == "__main__":
  embeddings = np.load("docs/embeddings.npy")
  filenames = pd.read_csv("docs/embedding_filenames.csv")["filename"].tolist()
  query_idx = 100                  
  results = find_similar(query_idx, embeddings, filenames)
  for fname, dist in results:
      print(f"{dist:.2f}  {fname}")

#picture seem not really correlating that much. TODO later different cluster or modl 
# 0.00  0000076_00616_d_0000003.jpg
# 14.11  0000313_04801_d_0000460.jpg
# 14.28  0000308_00801_d_0000309.jpg
# 14.37  0000349_04509_d_0000463.jpg
# 14.45  9999998_00395_d_0000347.jpg
# 14.71  0000076_03689_d_0000012.jpg