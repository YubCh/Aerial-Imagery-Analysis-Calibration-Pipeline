import cv2 as cv
from pathlib import Path


DATA_DIR = Path("data/VisDrone2019-DET-train/images")

def scan_dataset():
  img_path = sorted(DATA_DIR.glob("*.jpg"))
  print(f"Found: {len(img_path)} images")
  return img_path

def save_image(img_path):
  img = cv.imread(str(img_path))
  print(f"Loaded {img_path.name}, shape: {img.shape}")
  cv.imshow("Image", img)
  cv.waitKey(0)
  cv.destroyAllWindows()
  cv.imwrite('docs/sample_image.jpg', img)
  print('Saved copy: docs/sample_image.jpg')

if __name__ == "__main__":
  paths = scan_dataset()
  save_image(paths[0])