#!/bin/bash

#SBATCH --job-name=yolo_training

#SBATCH --account=project_2013501

#SBATCH --partition=gpu

#SBATCH --gres=gpu:v100:3

#SBATCH --cpus-per-task=8

#SBATCH --mem=64G

#SBATCH --time=12:00:00

#SBATCH --output=/scratch/project_2013501/yolo_train.log



# Load required modules

module purge

module load pytorch/2.0



# Print environment and Python version

echo "Running on $(hostname)"

echo "Python path: $(which python3)"

apptainer_wrapper exec python3 --version



# Run training inside the container

apptainer_wrapper exec python3 /scratch/project_2013501/train_yolo_model.py


