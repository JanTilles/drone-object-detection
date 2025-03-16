import os
import mlflow
from ultralytics import YOLO, settings

def setup_mlflow():

    mlflowStorage = "file:///C:/Users/JukkaPelto-aho/OneDrive - Pelto-aho Software Oy/Documents/yamk/AI-project/mlruns"

    """Set up MLflow experiment and logging."""
    settings.update({"mlflow": True})  # Enable MLflow logging
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "DIANA_YOLO_Training"
    os.environ["MLFLOW_RUN"] = "baseline_run"
    mlflow.set_tracking_uri(mlflowStorage)

    #print("Run: mlflow ui --backend-store-uri file:///C:/Users/Teemu/drone-object-detection/mlruns")
    print(f"Run: mlflow ui --backend-store-uri {mlflowStorage}")
    print("Open: http://127.0.0.1:5000 in your browser")

def train_model():
    """Train YOLO model with specified parameters."""
    model = YOLO('yolo11n.pt')  # Load pre-trained YOLO model

    train_params = {
        "data": "dataset_config.yaml",
        "epochs": 1,
        "batch": 16,  
        "imgsz": 640,  
        "device": "cpu",  # Use GPU 0, or -1 to run on CPU
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

    results = model.train(**train_params)

if __name__ == "__main__":
    setup_mlflow()
    train_model()
