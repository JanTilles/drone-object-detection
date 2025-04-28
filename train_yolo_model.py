import os
import mlflow
from ultralytics import YOLO, settings


def find_repo_root():
    """Find the root directory of the repository."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, '.git')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None

def setup_mlflow():
    """Set up MLflow experiment and logging."""
    #repo_root = find_repo_root()
    repo_root = "/scratch/project_2013501"
    if repo_root is None:
        raise RuntimeError("Repository root not found. Ensure the script is inside a Git repository.")

    settings.update({"mlflow": True})  # Enable MLflow logging
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "DIANA_YOLO_Training"
    os.environ["MLFLOW_RUN"] = "baseline_run"
    mlflow.set_tracking_uri(f"file:///{os.path.join(repo_root, 'mlruns')}")

    print(f"Run: mlflow ui --backend-store-uri file:///{os.path.join(repo_root, 'mlruns')}")
    print("Open: http://127.0.0.1:5000 in your browser")

def train_model():
    """Train YOLO model with specified parameters.""" 
    model = YOLO("yolov8m.pt") # Load pre-trained YOLO model


    train_params = {
        "data": "dataset_config.yaml",
        "epochs": 50,
        "batch": 16,  
        "imgsz": 1920,  
        "device": "0,1,2,3",  # Use GPU
        "project": "mlruns/DIANA",
        "name": "auto_twoclass_1920_Adam",
        "save": True,  
        "patience": 20,  
        "save_period": 10, 
        "workers": 4,  
        "classes": [0,1],
        "cos_lr": True,  
        "exist_ok":True,
        "optimizer": "adam",  
        "val": True,
        "lr0": 0.00846,
        "lrf": 0.00844,
        "momentum": 0.92663,
        "weight_decay": 0.00054,
        "warmup_epochs": 3.07784,
        "warmup_momentum": 0.87785,
        "box": 7.77595,
        "cls": 0.44652,
        "dfl": 1.29841,
        "hsv_h": 0.01474,
        "hsv_s": 0.69613,
        "hsv_v": 0.40177,
        "degrees": 0.0,
        "translate": 0.10486,
        "scale": 0.55003,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.49154,
        "bgr": 0.0,
        "mosaic": 0.93245,
        "mixup": 0.0,
        "copy_paste": 0.0
    }

    results = model.train(**train_params)

if __name__ == "__main__":
    setup_mlflow()
    train_model()

