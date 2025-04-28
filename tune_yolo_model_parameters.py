import os
from ultralytics import YOLO, settings



def train_model():
    """Train YOLO model with specified parameters.""" 
    model = YOLO("yolov8m.pt") # Load pre-trained YOLO model


    # Define search space
    search_space = {
        "lr0": (1e-5, 1e-1),
        "cls": (1.5, 4.0),
        "flipud": (0.3, 1.0)
    }

    train_params = {
        "data": "dataset_config.yaml",
        "epochs": 30,
        "iterations": 15,
        "batch": 16,
        "imgsz": 1920,
        "device": "0,1,2,3",
        "plots": False,
        "save": False,
        "val": False,
        "project": "tune_parameters",
        "name": "tune_parameters",
        "classes": [0,1],
        "cos_lr": True,
        "exist_ok":True,
        "optimizer": "AdamW"

    }

    model.tune(**train_params)
    

if __name__ == "__main__":
    train_model()
