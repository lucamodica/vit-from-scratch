"""
Contains functions to train the and evaluate the model.
"""

import torch as t
from torch import nn, optim
from dataclasses import dataclass
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from typing import Dict, List, Tuple
from utils import save_model

@dataclass
class TrainingArgs:
  epochs: int = 5
  batch_size: int = 32
  lr: float = 3e-4
  weight_decay: float = 1e-2
  device: str = "cuda" if t.cuda.is_available() else "cpu"
  model_save_path: str = "model.pth"
  
  use_wandb: bool = False
  wandb_project: str = "vit-from-scratch"
  wandb_name: str | None = None
  log_interval: int = 100
  
  

class ModelTrainer:
  def __init__(
    self, 
    model: nn.Module, 
    args: TrainingArgs, 
    train_loader: DataLoader, 
    val_loader: DataLoader,
    loss_fn: nn.Module = nn.CrossEntropyLoss(),
    optimizer: optim.Optimizer | None = None,
  ):
    super().__init__()
    
    self.model = model
    self.args = args
    self.device = args.device
    self.model.to(self.device)
    self.step = 0
    
    self.loss_fn = loss_fn
    self.optimizer = optimizer if optimizer is not None else optim.AdamW(
      self.model.parameters(), 
      lr=args.lr, 
      weight_decay=args.weight_decay
    )
    
    self.train_loader = train_loader
    self.val_loader = val_loader
    
  def training_step(self, X, y) -> Tuple[float, float]:
    """Performs a single training step (forward + backward pass) on a batch of data"""
    X, y = X.to(self.device), y.to(self.device)
    
    # Forward pass
    y_pred = self.model(X)
    
    # calc and log loss
    loss = self.loss_fn(y_pred, y)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    # compute accuracy
    y_pred_class = t.argmax(t.softmax(y_pred, dim=1), dim=1)
    train_acc = (y_pred_class == y).sum().item() / len(y_pred)
    
    self.step += 1
    if self.args.use_wandb:
      wandb.log({"train/loss": loss.item(), "train/accuracy": train_acc}, step=self.step)
    # elif self.step % self.args.log_interval == 0:
    #   print(f"Step {self.step}: train loss = {loss.item():.4f}, train accuracy = {train_acc:.4f}")
    
    return loss.item(), train_acc
  
  @t.inference_mode()
  def evaluate(self) -> Tuple[float, float]:
    """Evaluates the model on the validation set"""
    self.model.eval()
    val_loss, val_acc = 0.0, 0.0
    
    for X, y in tqdm(self.val_loader, desc="Evaluating"):
      X, y = X.to(self.device), y.to(self.device)
      
      y_pred = self.model(X)
      loss = self.loss_fn(y_pred, y)
      val_loss += loss.item()
      
      y_pred_class = t.argmax(t.softmax(y_pred, dim=1), dim=1)
      acc = (y_pred_class == y).sum().item() / len(y_pred)
      val_acc += acc
          
    val_loss /= len(self.val_loader)
    val_acc /= len(self.val_loader)
    
    if self.args.use_wandb:
      wandb.log({"val/loss": val_loss, "val/accuracy": val_acc}, step=self.step)
    else:
      print(f"Validation: loss = {val_loss:.4f}, accuracy = {val_acc:.4f}")
    
    return val_loss, val_acc
  
  def train(self) -> Dict[str, List[float]]:
    """Trains the model for the specified number of epochs"""
    
    if self.args.use_wandb:
      wandb.init(
        project=self.args.wandb_project, 
        name=self.args.wandb_name, 
        config=vars(self.args)
      )
      wandb.watch(self.model, log="all", log_freq=self.args.log_interval)
      
    results = {
      "train_loss": [],
      "train_acc": [],
      "val_loss": [],
      "val_acc": []
    }
    
    val_loss, val_acc = self.evaluate()  # eval before training to get baseline
    results["val_loss"].append(val_loss)
    results["val_acc"].append(val_acc)
    
    for epoch in range(self.args.epochs):
      self.model.train()
      
      pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
      train_loss, train_acc = 0.0, 0.0
      
      for X, y in pbar:
        loss, acc = self.training_step(X, y)
        pbar.set_postfix({"loss": train_loss, "step": self.step})
      
        train_loss += loss
        train_acc += acc
        
      train_loss /= len(self.train_loader)
      train_acc /= len(self.train_loader)
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      
      val_loss, val_acc = self.evaluate()
      pbar.set_postfix(
          loss=f"{val_loss:.3f}", accuracy=f"{val_acc:.2f}", step=f"{self.step=:06}"
      )
      results["val_loss"].append(val_loss)
      results["val_acc"].append(val_acc)
      
    
    # save model
    save_model(self.model, target_dir="../models", model_name=self.args.model_save_path)
      
    if self.args.use_wandb:
      wandb.finish()
    
    return results
    
  
# test the class with a dummy model and data
if __name__ == "__main__":
  from torch.utils.data import DataLoader, TensorDataset
  
  # Create a dummy dataset
  X_train = t.randn(100, 3, 32, 32)
  y_train = t.randint(0, 10, (100,))
  X_val = t.randn(20, 3, 32, 32)
  y_val = t.randint(0, 10, (20,))
  
  train_dataset = TensorDataset(X_train, y_train)
  val_dataset = TensorDataset(X_val, y_val)
  
  train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=16)
  
  # Create a simple model
  class SimpleCNN(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
      self.pool = nn.MaxPool2d(2, 2)
      self.fc1 = nn.Linear(16 * 16 * 16, 10)
      
    def forward(self, x):
      x = self.pool(t.relu(self.conv1(x)))
      x = x.view(-1, 16 * 16 * 16)
      x = self.fc1(x)
      return x
  
  model = SimpleCNN()
  
  args = TrainingArgs(epochs=10, use_wandb=False)
  
  trainer = ModelTrainer(model, args, train_loader, val_loader)
  results = trainer.train()
  
  print("Training complete. Results:", results)