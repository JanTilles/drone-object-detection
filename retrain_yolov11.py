import os
import mlflow
from ultralytics import YOLO

def find_repo_root():
    """Find the root directory of the repository."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):  # Stop at root '/'
        if os.path.exists(os.path.join(current_dir, '.git')):  # Found .git folder
            return current_dir
        current_dir = os.path.dirname(current_dir)  # Move up one level
    return None  # If no .git is found

def setup_mlflow():
    """Set up MLflow experiment and logging."""
    repo_root = find_repo_root()
    if repo_root is None:
        raise RuntimeError("Repository root not found. Ensure the script is inside a Git repository.")

    mlflow.set_tracking_uri(f"file:///{os.path.join(repo_root, 'mlruns')}")

    print(f"Run: mlflow ui --backend-store-uri file:///{os.path.join(repo_root, 'mlruns')}")
    print("Open: http://127.0.0.1:5000 in your browser")

def train_model():
    """Train YOLO model with specified parameters."""
    repo_root = find_repo_root()
    if repo_root is None:
        raise RuntimeError("Repository root not found. Ensure the script is inside a Git repository.")

    # Define absolute path to dataset_config.yaml
    dataset_yaml = os.path.join(repo_root, "dataset_config.yaml")

    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"Dataset configuration file not found at: {dataset_yaml}")

    print(f"Using dataset YAML: {dataset_yaml}")

    # Load YOLO model
    model = YOLO('yolo11n.pt')

    # Define training parameters
    train_params = {
        "data": dataset_yaml,  # Use absolute path
        "epochs": 1,
        "batch": 16,  
        "imgsz": 640,  
        "device": "cpu",  # Use CPU, if GPU is not available. Use '0' for GPU
        "project": "mlruns/DIANA",
        "name": "baseline_run",
        "save": True,  
        "patience": 20,  
        "save_period": 10, 
        "workers": 8,  
        "lr0": 0.01,  
        "lrf": 0.001,  
        "momentum": 0.937,  
        "weight_decay": 0.0005,  
        "warmup_epochs": 5, 
        "cos_lr": True,  
        "optimizer": "auto",  
        "pretrained": True,  
        "verbose": True, 
        "val": True  
    }

    # Train model
    results = model.train(**train_params)

if __name__ == "__main__":
    setup_mlflow()
    train_model()
