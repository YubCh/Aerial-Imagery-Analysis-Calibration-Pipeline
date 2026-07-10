import cv2 as cv
import numpy as np
import pandas as pd
import math
from pathlib import Path
from src.s1_datascan.scan import scan_dataset


def to_grayscale(img):
  return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def compute_brightness(gray):
  return float(np.mean(gray))

def compute_contrast(gray):
  return float(np.std(gray))

def compute_blur_score(gray):
  return float(cv.Laplacian(gray, cv.CV_64F).var())

def compute_histogram(gray):
  hist = cv.calcHist(
    [gray],
    [0],
    None,
    [256], 
    [0, 256])
  return hist.flatten()

def build_stats_table(image_paths):
  rows = []
  for i, path in enumerate(image_paths):
    img = cv.imread(str(path))
    gray = to_grayscale(img)
    rows.append({
      "filename": path.name,
      "width": img.shape[1],
      "height": img.shape[0],
      "brightness": compute_brightness(gray),
      "contrast": compute_contrast(gray),
      "blur_score": compute_blur_score(gray)
    })
    if i % math.floor(len(image_paths)/100) == 0:
      print(f"processed {100 * (i / len(image_paths))} %")
  return pd.DataFrame(rows)



if __name__ == "__main__":
  paths = scan_dataset()
  df = build_stats_table(paths)
  df.to_csv('docs/image_stats.csv', index=False)
  print(df.describe())
  
  # img = cv.imread('data/VisDrone2019-DET-train/images/0000008_00889_d_0000039.jpg')
  # gray = to_grayscale(img)
  # print(f"Brightness: {compute_brightness(gray)}")
  # print(f"Blur Score: {compute_blur_score(gray)}")
  # print(f"Contrast: {compute_contrast(gray)}")
  # print(f"Histogram: \n {pd.DataFrame({'pixel': range(256), "count": compute_histogram(gray)})}")
  # #quick test
  # sharp = cv.Laplacian(gray, cv.CV_64F).var()
  # blurred = cv.GaussianBlur(gray, (9, 9), 0)
  # blurry = cv.Laplacian(blurred, cv.CV_64F).var()
  # print(f"Sharp: {sharp}")
  # print(f"Blurred: {blurry}")