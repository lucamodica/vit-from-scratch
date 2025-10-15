"""
Data setup module for the project.

This module is responsible for setting up the necessary data structures
and loading data from various sources, mainly through DataLoader classes.
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch as t

NUM_WORKERS = os.cpu_count() or 1

def create_dataloaders(
  train_dir: str,
  test_dir: str,
  batch_size: int = 32,
  num_workers: int = NUM_WORKERS,
  train_data_percent: float = 1.0,
  transforms: transforms.Compose = transforms.Compose([transforms.ToTensor()])
):
  """
  Create DataLoader objects for training and validation datasets.
  
  Args:
      train_dir (str): Directory path for the training dataset.
      test_dir (str): Directory path for the test dataset.
      batch_size (int): Number of samples per batch to load. Default is 32.
      num_workers (int): Number of subprocesses to use for data loading. Default is number of CPU cores.
      transforms (transforms.Compose): Transformations to apply to the data. Default is ToTensor.
      
  """
  train_data = datasets.ImageFolder(root=train_dir, transform=transforms)
  test_data = datasets.ImageFolder(root=test_dir, transform=transforms)
  
  # get class names
  class_names = train_data.classes
  print(f"Class names: {class_names}")
  
  # optionally subsample the training data
  if train_data_percent < 1.0:
    num_train_samples = int(len(train_data) * train_data_percent)
    train_data, _ = t.utils.data.random_split(train_data, [num_train_samples, len(train_data) - num_train_samples])
    print(f"Subsampled training data to {num_train_samples} samples.")
  
  train_loader = DataLoader(
    dataset=train_data, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    pin_memory=True
  )
  
  test_loader = DataLoader(
    dataset=test_data, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers,
    pin_memory=True
  )
  
  return train_loader, test_loader, class_names