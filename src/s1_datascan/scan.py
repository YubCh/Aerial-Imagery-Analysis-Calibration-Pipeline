import cv2 as cv
from pathlib import Path


DATA_DIR = Path("data/VisDrone2019-DET-train/images")
ANNO_DIR = Path("data/VisDrone2019-DET-train/annotations")

def scan_dataset():
  img_path = sorted(DATA_DIR.glob("*.jpg"))
  print(f"Found: {len(img_path)} images")
  return img_path

def save_image(img_path):
  img = cv.imread(str(img_path))
  print(f"Loaded Image: {img_path.name}, shape: {img.shape}")
  # cv.imshow("Image", img)
  # cv.waitKey(0)
  # cv.destroyAllWindows()
  cv.imwrite('docs/sample_image.jpg', img)
  print('Saved copy: docs/sample_image.jpg')

def draw_annotations(image_path):
  img = cv.imread(str(image_path))
  anno_path = ANNO_DIR / image_path.name.replace(".jpg", ".txt")
  #VisDrone annotations:                                                  <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion> x = left position, y = top position, w = width, h = height
  with open(anno_path) as f:
    for line in f:
      parts = line.strip().split(",")
      x,y,w,h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
      cv.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 0, 0), #colr
        1 # thickness
      )
  # cv.imshow("Image", img)
  # cv.waitKey(0)
  # cv.destroyAllWindows()    
  cv.imwrite('docs/annotated_sample.jpg',img)
  print('Saved copy: docs/annotated_sample.jpg')


if __name__ == "__main__":
  paths = scan_dataset()
  save_image(paths[0])
  draw_annotations(paths[0])