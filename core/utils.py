from pathlib import Path
from typing import Dict, List
import torch as t
from torch import nn
import requests
import zipfile
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def setup_working_directory() -> None:
    """Sets up the repo root as working directory."""
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)
    print(f"[INFO] Working directory set to: {repo_root}")

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    t.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    t.cuda.manual_seed(seed)
    
def save_model(model: nn.Module, target_dir: str, model_name: str) -> None:
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  t.save(obj=model.state_dict(), f=model_save_path)
  
def download_data(source: str, destination: str, remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("../data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path

def plot_loss_curves(results: Dict[str, List[float]]) -> None:
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "val_loss": [...],
             "val_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and validation)
    loss = results['train_loss']
    val_loss = results['val_loss']

    # Get the accuracy values of the results dictionary (training and validation)
    accuracy = results['train_acc']
    val_accuracy = results['val_acc']
    
    # if the length of train and val array mismatch, that's
    # because we evaluate before training to get a baseline
    # take the values after the baseline in that case
    if len(loss) != len(val_loss):
        val_loss = val_loss[1:]
        val_accuracy = val_accuracy[1:]

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

def pred_and_plot_image(
    model: nn.Module, 
    img_path: Path, 
    class_names: List[str],
    transform: transforms.Compose = None,
    img_size: int = (224, 224),
    device="cpu"
) -> None:
    """Makes a prediction on a single image file and plots the image with the prediction.

    Args:
        model (nn.Module): A trained PyTorch model for making predictions.
        img_path (Path): Path to a single image file.
        class_names (List[str]): A list of class names for mapping labels to names.

    Example usage:
        pred_and_plot_image(model=model_0,
                            img_path=Path("some_image.jpg"),
                            class_names=class_names)
    """
    model = model.to(device)
    model.eval()
    
    # Load in the image and convert to a tensor
    img = Image.open(img_path).convert("RGB")
    if not transform:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    with t.inference_mode():
        # adjust image tensor dims
        img_batch = transform(img).unsqueeze(0).to(device)
        
        preds = model(img_batch)
        
        # Move image back to cpu and denormalize for plotting
        plt.imshow(img)
        plt.title(f"Pred: {class_names[preds.argmax()]} | Prob: {t.softmax(preds, dim=1).max():.3f}")
        plt.axis('off')
        plt.show()
        

# test the utils functions
if __name__ == "__main__":
    setup_working_directory()