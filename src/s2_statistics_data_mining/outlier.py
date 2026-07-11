import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest



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



if __name__ == "__main__":
    df = pd.read_csv("docs/image_stats.csv")
    columns = ["brightness", "contrast", "blur_score"]

    df = zscore(df, columns)
    z_flags = zscore_outliers(df, columns)
    print(f"z-score outliers: {len(z_flags)}")
    print(z_flags.head())


    print(isolation_forest_outliers(df, columns).head())

  