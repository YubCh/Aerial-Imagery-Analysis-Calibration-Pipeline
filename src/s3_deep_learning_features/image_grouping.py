import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import shutil
from pathlib import Path



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


if __name__ == "__main__":
  embeddings = np.load("docs/embeddings.npy")
  filenames = pd.read_csv("docs/embedding_filenames.csv")["filename"].tolist()
  cluster_ids = cluster_images(embeddings, 30)
  export_cluster_samples(cluster_ids, filenames, 0)
  export_cluster_samples(cluster_ids, filenames, 1)
  export_cluster_samples(cluster_ids, filenames, 4)   