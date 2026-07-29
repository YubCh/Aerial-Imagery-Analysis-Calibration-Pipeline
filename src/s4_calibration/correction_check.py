import cv2
import numpy as np
from src.s4_calibration.image_correction import calibrate_camera, undistort_image ,normalize_exposure
from src.s3_deep_learning_features.image_classification import load_model, extract_embedding
from pathlib import Path

def make_dark(img, factor=0.5):
    return (img * factor).astype(np.uint8)   

def make_bright(img, factor=1.5):
    return np.clip(img * factor, 0, 255).astype(np.uint8)


if __name__ == "__main__":
  folder = Path('data/VisDrone2019-DET-train/images')
  model = load_model()
  befores = []
  afters = []

  for image_path in list(folder.glob("*.jpg"))[:20]:
    img = cv2.imread(image_path)

    dark = make_dark(img)
    bright = make_bright(img)
    
    emb_dark = extract_embedding(model, dark)
    emb_bright = extract_embedding(model, bright)
    distance_before = np.linalg.norm(emb_dark - emb_bright)

    dark_normalized = normalize_exposure(dark)
    bright_normalized = normalize_exposure(bright)
    
    emb_dark_normalized = extract_embedding(model, dark_normalized)
    emb_bright_nomalized = extract_embedding(model, bright_normalized)
    distance_after = np.linalg.norm(emb_dark_normalized - emb_bright_nomalized)
    print("distance before correction:", distance_before)
    print("distance after correction: ", distance_after)

    befores.append(distance_before)
    afters.append(distance_after)

  befores = np.array(befores)
  afters = np.array(afters)
  worked = np.sum(afters < befores)           
  print(f"average before: {befores.mean():.2f}")
  print(f"average after:  {afters.mean():.2f}")
  print(f"helped in {worked}/20 cases")


 
 