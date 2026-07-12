import pandas as pd
from sklearn.ensemble import IsolationForest
import shutil
from pathlib import Path


def zscore(df, columns):
  for col in columns:
    df[f"{col}_zscore"] = (df[col] - df[col].mean()) / df[col].std()
  return df


def zscore_outliers(df, columns, threshhold = 3.0):
  outliers = []
  for col in columns:
    outlier_rows = df[
      df[f"{col}_zscore"].abs() > threshhold
    ]

    for _, row in outlier_rows.iterrows():
      outliers.append({
        "filename": row["filename"],
        "method": "zscore",
        "metric": col,
        "score": float(row[f"{col}_zscore"])
      })

  return pd.DataFrame(outliers)

def isolation_forest_outliers(df, columns, contamination = 0.01):
  model = IsolationForest(contamination=contamination, random_state=42)
  prediction = model.fit_predict(df[columns])
  scores = model.decision_function(df[columns])
  outliers = df[prediction == -1]

  return pd.DataFrame({
    'filename': outliers['filename'],
    'method': 'IsolationForest',
    'metric': 'multivariate',
    'score': scores[prediction == -1]
  })

 
def export_outlier_head(outliers, n=10):
  out_dir = Path("docs/outlier_samples")
  out_dir.mkdir(exist_ok=True)
  src_dir = Path("data/VisDrone2019-DET-train/images")
  for filename in outliers['filename'].head(n):
    shutil.copy(src_dir / filename, out_dir / filename)
  print('saved outliers at ', out_dir)

if __name__ == "__main__":
    df = pd.read_csv("docs/image_stats.csv")
    columns = ["brightness", "contrast", "blur_score"]
 
  
    df =  zscore(df, columns)
    all_flags = pd.concat([zscore_outliers(df, columns), isolation_forest_outliers(df, columns)])
    all_flags.to_csv("docs/outliers.csv", index=False)
    print(f"total flags saved: {len(all_flags)}")