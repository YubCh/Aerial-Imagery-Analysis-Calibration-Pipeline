import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms


device = "mps" if torch.backends.mps.is_available else "cpu"


def load_model():
  model = models.resnet18(weights="IMAGET1K_V1")
  model.eval()
  model.to(device)
  print(f"model ready to use on {device}")
  return model