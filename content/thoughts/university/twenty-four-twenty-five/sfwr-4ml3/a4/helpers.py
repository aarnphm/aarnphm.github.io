from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import timm  # for loading EfficientNetV2
from tqdm import tqdm
import os
from datetime import datetime
from safetensors.torch import save_file, load_file


class TrainingHistory:
  def __init__(self):
    self.train_losses = []
    self.train_accuracies = []
    self.val_losses = []
    self.val_accuracies = []
    self.epochs = []

  def update(self, epoch, train_loss, train_acc, val_loss, val_acc):
    self.epochs.append(epoch)
    self.train_losses.append(train_loss)
    self.train_accuracies.append(train_acc)
    self.val_losses.append(val_loss)
    self.val_accuracies.append(val_acc)

  def save(self, filepath):
    history = {
      'epochs': self.epochs,
      'train_loss': self.train_losses,
      'train_acc': self.train_accuracies,
      'val_loss': self.val_losses,
      'val_acc': self.val_accuracies,
    }
    np.save(filepath, history)

  @classmethod
  def load(cls, filepath):
    history = cls()
    loaded = np.load(filepath, allow_pickle=True).item()
    history.epochs = loaded['epochs']
    history.train_losses = loaded['train_loss']
    history.train_accuracies = loaded['train_acc']
    history.val_losses = loaded['val_loss']
    history.val_accuracies = loaded['val_acc']
    return history


class PretrainedMixin:
  @classmethod
  def from_pretrained(cls, filepath, device='cuda'):
    model = cls().to(device)
    # Load model weights
    load_model(model, filepath)
    model.eval()
    history = None
    # Try to load history if it exists
    history_path = filepath.replace('.safetensors', '_history.npy')
    if os.path.exists(history_path):
      history = TrainingHistory.load(history_path)
    return model, history

  def save_pretrained(self, base_path='./model', history=None):
    os.makedirs(base_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create filenames with model name and timestamp
    model_filename = f'{self.__class__.__qualname__}_{self.variants}_{timestamp}.safetensors'
    history_filename = f'{self.__class__.__qualname__}_{self.variants}_{timestamp}_history.npy'

    model_filepath = os.path.join(base_path, model_filename)
    history_filepath = os.path.join(base_path, history_filename)

    # Save the model
    save_model(self, model_filepath)

    # Save the history if provided
    if history is not None:
      history.save(history_filepath)
      print(f'Model and history saved to {model_filepath} and {history_filepath}')
    else:
      print(f'Model saved to {model_filepath}')

    return model_filepath


class MetricsTracker:
  def __init__(self):
    self.reset()

  def reset(self):
    self.running_loss = 0.0
    self.correct = 0
    self.total = 0
    self.batch_count = 0

  def update(self, loss, predicted, labels):
    self.running_loss += loss
    self.batch_count += 1
    self.total += labels.size(0)
    self.correct += predicted.eq(labels).sum().item()

  @property
  def avg_loss(self): return self.running_loss / self.batch_count if self.batch_count > 0 else 0

  @property
  def accuracy(self): return 100.0 * self.correct / self.total if self.total > 0 else 0
