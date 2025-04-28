import os
from ultralytics import YOLO, settings



def train_model():
    """Train YOLO model with specified parameters.""" 
    model = YOLO("yolov8m.pt") # Load pre-trained YOLO model


    # Define search space
    search_space = {
        "lr0": (1e-5, 1e-1),
        "box": (0.0, 45.0),
    }
    #            "space": search_space,

    train_params = {
        "data": "dataset_config.yaml",
        "epochs": 1,
        "iterations": 10,
        "batch": 16,  
        "imgsz": 1920,
        "device": "cpu",
        "plots": False,
        "save": False,
        "val": False,
        "project": "tune_parameters",
        "name": "tune_parameters",
        "classes": [0,1],
        "cos_lr": True,  
        "exist_ok":True,
        "optimizer": "AdamW",
        "time": 11
    }

    model.tune(**train_params)
    

if __name__ == "__main__":
    train_model()