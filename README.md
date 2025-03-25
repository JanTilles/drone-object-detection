# 🧠 DIANA CNN Training on Puhti (CSC HPC Finland)

> **Note:** The DIANA dataset is not included in this repository. Additionally, its original COCO format has been modified to the structure presented here before training.

This project contains a TensorFlow-based Convolutional Neural Network (CNN) pipeline to train a deep learning model on the [DIANA drone dataset](https://www.kaggle.com/datasets/aminmajd/diana-drone-imagery-for-archipelago-navigation), optimized for **multi-GPU training on Puhti**, the CSC supercomputer in Finland.
---

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

`train_model.sh` example (requesting 4 GPUs, 48G memory):

```bash
#!/bin/bash
#SBATCH --job-name=yolo_training
#SBATCH --account=project_2013587
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_<ID>/<USERNAME>/yolo_train.log

# Load required modules
module purge
module load pytorch/2.0

# Print environment and Python version
echo "Running on $(hostname)"
echo "Python path: $(which python3)"
apptainer_wrapper exec python3 --version

# Run training inside the container
apptainer_wrapper exec python3 /scratch/project_<ID>/<USERNAME>/train_model.py
```

---

### 3. Submit the Training Job

```bash
cd /scratch/project_<ID>
sbatch train_model.sh
```

---

### 4. Monitor Your Job

```bash
squeue -u <USERNAME>                  # View job queue
tail -f train_output.log            # Follow training log
seff <JOB_ID>                       # Summary after run (GPU/CPU usage)
```

---

## ✅ Output

- `best_model.pt`: Best model saved during training (based on validation loss).
- `last_model.pt`: Model saved at the final epoch.
- `training_metrics.json`: JSON file containing training and validation metrics (e.g., loss, accuracy).
- `training_curves.png`: Plot of training and validation loss/accuracy over epochs.
- `yolo_train.log`: Log file containing detailed training output.

---

## 📬 Author

Created by **Jan Tilles**  
Contact: [jan.tilles@example.com](mailto:jan.tilles@example.com)  
For more on Puhti: [CSC Docs](https://docs.csc.fi)

---
