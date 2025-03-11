import os
from ultralytics import YOLO

def train_yolo(data_config, model='yolov5s', epochs=100):
    # Initialize the model
    model = YOLO(model)
    
    # Train the model
    model.train(data=data_config, epochs=epochs)

if __name__ == '__main__':
    data_config = 'c:/Users/extjtilles/Documents/Work/Python/drone-object-detection/dataset_config.yaml'
    train_yolo(data_config)
