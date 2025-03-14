# Drone Imagery for Archipelago Navigation and Analysis (DIANA)

## Project Overview
This project focuses on developing an accurate and lightweight computer vision algorithm for detecting objects in maritime environments using drone imagery. We're utilizing the DIANA dataset, which consists of annotated drone images collected from Finland's southwestern archipelago. Our primary objective is to optimize detection accuracy while ensuring computational efficiency using the YOLO (You Only Look Once) model architecture.

## Dataset Description
The DIANA dataset was collected in the summer of 2020 in Finland's southwestern archipelago. This area was chosen for its environmental complexity, which contrasts with the simpler backgrounds of open-sea datasets.

### Key Features
- **High-resolution images**: 3840×2160 pixels
- **Volume**: 17,758 images with 353,141 object instances
- **Object classes**: 5 maritime object types
  - motor_boat
  - sailing_boat
  - ship
  - sea_mark
  - floating_object
- **Capture altitude**: 8 to 120 meters
- **Diverse scenarios**: Various weather conditions, sun orientations, heights, angles, and vessel types
- **Annotation format**: COCO format (converted to YOLO format for training)
- **Split ratio**: 70% training, 15% validation, 15% testing

### Environmental Complexity
While open-sea datasets mostly feature water and sky, DIANA includes diverse scenes with islands, forests, buildings, and vehicles, making it more challenging for object detection algorithms.

## Project Structure
```
drone-object-detection/
│── .git/
│── .venv/
│── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── test/
│   │   ├── val/
│   ├── labels/
│       ├── train/
│       ├── test/
│       ├── val/
│── helpers/
│   ├── convert_coco_to_yolo.py
│── .gitignore
│── dataset_config.yaml
│── README.md
│── requirements.txt
│── retrain_yolov11.py
```

## Model Information

**[WORK IN PROGRESS]**

This project uses YOLO as the base model with MLflow for experiment tracking. 

## Usage

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/username/drone-object-detection.git
cd drone-object-detection

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

**Note:** The dataset is not included in this repository due to its large size. However, the DIANA dataset has been converted from COCO to YOLO format for the project. This conversion process was completed using the `convert_coco_to_yolo.py` script in the helpers directory.

For reference, if starting with the original COCO format annotations, the conversion would require the following directory structure:
```
drone-object-detection/
│── images/             # Original COCO format images
│── annotations/        # Original COCO format annotation files
│── helpers/
│   ├── convert_coco_to_yolo.py
```

And then running:
```bash
python helpers/convert_coco_to_yolo.py
```

This script handles:
1. Cleaning annotation files
2. Incrementing category IDs as needed
3. Converting COCO annotations to YOLO format
4. Organizing images into train/val/test directories
5. Creating the final dataset structure

### Training the Model
```bash
python retrain_yolov11.py
```

### Monitoring Training
After starting training:
1. Run: `mlflow ui --backend-store-uri file:///path/to/drone-object-detection/mlruns`
2. Open: http://127.0.0.1:5000 in your browser

## Project Goals
- Develop a robust object detection model for maritime environments
- Optimize for both accuracy and computational efficiency
- Create a model that generalizes well across diverse environmental conditions
- Document the entire machine learning pipeline from data preprocessing to model evaluation