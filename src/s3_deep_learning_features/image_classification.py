import cv2
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from src.s1_datascan.scan import scan_dataset


device = "mps" if torch.backends.mps.is_available else "cpu"


def load_model():
  model = models.resnet18(weights="IMAGENET1K_V1")
  model.fc = torch.nn.Identity()
  model.eval()
  model.to(device)
  print(f"model ready to use on {device}")
  return model


preprocess = transforms.Compose([
  transforms.ToPILImage(),
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(), # to rgb to0 -1
  transforms.Normalize(mean=[0.485, 0.456, 0.406], #StandardScaler
                       std= [0.229, 0.224, 0.225]),
])

def extract_embedding(model, img_cv):
  img_rgb= cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
  tensor = preprocess(img_rgb).unsqueeze(0).to(device)
  with torch.no_grad():
    embedding = model(tensor) #final execution taking tensor to the whole modl
  return embedding.squeeze().cpu().numpy()


 

def build_all_embeddings(image_paths, model):
  res = []
  filenames = []
  for i,picture in enumerate(image_paths):
    img_cv = cv2.imread(str(picture))     
    emb = extract_embedding(model, img_cv)
    res.append(emb)
    filenames.append(picture.name)
  embeddings = np.stack(res)
  np.save("docs/embeddings.npy", embeddings)
  pd.DataFrame({"filename": filenames}).to_csv("docs/embedding_filenames.csv", index=False)
  return embeddings


if __name__ == "__main__":
  model = load_model()
  paths = scan_dataset()    
  build_all_embeddings(paths, model)
 