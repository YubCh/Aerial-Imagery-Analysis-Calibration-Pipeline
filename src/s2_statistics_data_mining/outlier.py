import pandas as pd
import numpy as np



def zscore(df, columns):
  for col in columns:
    df[f"{col}_zscore"] = (df[col] - df[col].mean()) / df[col].std()
  return df


def z_score_outliers(df, columns, threshhold = 3.0):
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

  return df.DataFrame(outliers)
