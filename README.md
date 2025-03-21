# 🧠 DIANA CNN Training on Puhti (CSC HPC Finland)

> **Note:** The DIANA dataset is not included in this repository. Additionally, its original COCO format has been modified to the structure presented here before training.

This project contains a TensorFlow-based Convolutional Neural Network (CNN) pipeline to train a deep learning model on the [DIANA drone dataset](https://www.kaggle.com/datasets/aminmajd/diana-drone-imagery-for-archipelago-navigation), optimized for **multi-GPU training on Puhti**, the CSC supercomputer in Finland.

---

## 📁 Folder Structure

```
.
├── train_model.py                # Main Python training script
├── train_model.sh                # SLURM batch job for Puhti
├── DIANA/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── best_model.keras             # Automatically saved best model (val_loss)
├── diana_trained_model.keras    # Last-epoch model (may not be best)
├── training_curves.png          # Training vs validation plots
└── README.md
```

---

## ⚙️ Requirements on Puhti

- **TensorFlow**: Load the module with GPU support
- **CUDA**: Version 12.6 or whichever is needed
- **Apptainer (Singularity)** is used by CSC internally when loading modules

---

## 🧪 Setup Instructions (Puhti)

### 1. Upload or extract your dataset to scratch
Your image structure must follow:
```
DIANA/images/train/*.jpg
DIANA/images/val/*.jpg
DIANA/images/test/*.jpg
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
#SBATCH --job-name=diana_tf
#SBATCH --account=project_2013587
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_2013587/tillesja/train_output.log

module purge
module load tensorflow/2.18
module load cuda/12.6.0

apptainer_wrapper exec python3 /scratch/project_2013587/train_model.py
```

---

### 3. Submit the Training Job

```bash
cd /scratch/project_2013587
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

- `best_model.keras`: Best model saved during training (based on `val_loss`)
- `diana_trained_model.keras`: Last model (final epoch)
- `training_curves.png`: Accuracy and loss plots

---

## 💡 Notes

- The model uses **ResNet50 (frozen or fine-tuned)** for feature extraction
- Training is distributed using `tf.distribute.MirroredStrategy` (multi-GPU)
- Images are streamed using the efficient `tf.data` pipeline

---

## 🔧 Future Ideas

- Add TensorBoard logging
- Add COCO-style evaluation metrics
- Fine-tune ResNet layers for better accuracy

---

## 📬 Author

Created by **Jan Tilles**  
Contact: [jan.tilles@example.com](mailto:jan.tilles@example.com)  
For more on Puhti: [CSC Docs](https://docs.csc.fi)

---
