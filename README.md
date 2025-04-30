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
│   ├── show_image_with_labels.py
│   ├── split_diana_images.py
│── .gitignore
│── dataset_config.yaml
│── README.md
│── requirements.txt
│── slurm_train_model.sh
│── train_yolo_model.py
│── tune_yolo_model_parameters.py
```

## Model Information


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

## Project Goals
- Develop a robust object detection model for maritime environments
- Optimize for both accuracy and computational efficiency
- Create a model that generalizes well across diverse environmental conditions
- Document the entire machine learning pipeline from data preprocessing to model evaluation

## 🧪 Setup Instructions (Puhti)

### 1. Upload or extract your dataset to scratch
Your image structure must follow:
```
DIANA/images/train/*.jpg
DIANA/images/val/*.jpg
DIANA/images/test/*.jpg
DIANA/labels/train/*.txt
DIANA/labels/val/*.txt
DIANA/labels/test/*.txt
DIANA/annotations/train.json
DIANA/annotations/val.json
DIANA/annotations/test.json
```

📍 Place under:
```
/scratch/project_XXXXXXX/<username>/DIANA/
```

---

### 2. Edit the SLURM Batch Script

`slurm_train_model.sh` example (requesting 4 GPUs, 48G memory):

```bash
#!/bin/bash


#SBATCH --account=project_2013501

#SBATCH --partition=gpu

#SBATCH --gres=gpu:v100:4

#SBATCH --cpus-per-task=8

#SBATCH --mem=64G

#SBATCH --time=10:00:00

#SBATCH --output=/scratch/project_xxxxxxx/yolo_train.log

#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load required modules

module --force purge

module load python-data
export PYTHONUSERBASE=/scratch/project_xxxxxxx/my-python-env
# Print environment and Python version

echo "Running on $(hostname)"

echo "Python path: $(which python3)"

set -xv
python3 $*

```

---

### 3. Submit the Training Job

```bash
cd /scratch/project_<ID>
sbatch slurm_train_model.sh train_yolo_model.py
```

---

### 4. Monitor Your Job

```bash
squeue -u <USERNAME>                  # View job queue
tail -f yolo_train.log            # Follow training log
seff <JOB_ID>                       # Summary after run (GPU/CPU usage)
```

---

## ✅ Output

- `best_model.pt`: Best model saved during training (based on validation loss).
- `last_model.pt`: Model saved at the final epoch.
- `yolo_train.log`: Log file containing detailed training output.
- various metrics and graphs

---