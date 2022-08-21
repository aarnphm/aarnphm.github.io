from __future__ import annotations

import os, inspect
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from tqdm import tqdm
from safetensors.torch import save_model, load_model

from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights

from helpers import PretrainedMixin

# Define CIFAR-100 mean and std
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

def test(model, test_csv_path='./test.csv', device='cuda', ncols=100):
  # Define transform for test data
  test_transform = transforms.Compose([
      transforms.Resize(224),
      transforms.ToTensor(),
      transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
  ])

  df = pd.read_csv(test_csv_path)
  predictions = []

  model.eval()
  with torch.no_grad(), tqdm(df.iterrows(), total=len(df), desc='inference', ncols=ncols) as pbar:
    for index, row in pbar:
      # Extract pixel data and convert to numpy array
      pixel_data = row[[f'pixel_{i}' for i in range(1, 3073)]].values.astype(np.float32)
      # Reshape to (3, 32, 32)
      image = pixel_data.reshape(3, 32, 32)
      # Convert to torch tensor
      image_tensor = torch.tensor(image)
      # Create PIL Image from tensor for resizing
      image_np = image_tensor.numpy().transpose(1, 2, 0)
      image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
      # Apply transforms
      image_tensor = test_transform(image_pil)
      # Add batch dimension and move to device
      image_tensor = image_tensor.unsqueeze(0).to(device)
      # Predict
      output = model(image_tensor)
      _, predicted = output.max(1)
      predictions.append(predicted.item())

  submission_df = pd.DataFrame({ 'ID': range(0, len(predictions)), 'LABEL': predictions })

  return submission_df

class EfficientNetV2Classifier(nn.Module, PretrainedMixin):
  variants="s".upper()

  def __init__(self, num_classes=100, variant="s"):
    super(EfficientNetV2Classifier, self).__init__()
    if variant == "s"   : weights=EfficientNet_V2_S_Weights.DEFAULT
    elif variant  == "m": weights=EfficientNet_V2_M_Weights.DEFAULT
    elif variant == "l" : weights=EfficientNet_V2_L_Weights.DEFAULT
    self.efficientnet = efficientnet_v2_s(weights=weights)
    self.variants = variant.upper()

    # patch image1k with cifar100
    num_features = self.efficientnet.classifier[1].in_features
    self.efficientnet.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_features, num_classes))

  def forward(self, x): return self.efficientnet(x)


if __name__ == "__main__":
  test_csv_path = 'test.csv'
  device = "cuda" if torch.cuda.is_available() else "cpu"

  model, history = EfficientNetV2Classifier.from_pretrained("./model/EfficientNetV2Classifier_20241204_034642.safetensors", device)

  submission_df = test(model, test_csv_path=test_csv_path, device=device)
  # Save predictions
  submission_df.to_csv('submission.csv', index=False)
