#!/bin/bash

#SBATCH --job-name=yolo_training

#SBATCH --account=project_2013587

#SBATCH --partition=gpu

#SBATCH --gres=gpu:v100:1

#SBATCH --cpus-per-task=4

#SBATCH --mem=48G

#SBATCH --time=04:00:00

#SBATCH --output=/scratch/project_2013587/tillesja/yolo_train.log



# Load required modules

module purge

module load pytorch/2.0



# Print environment and Python version

echo "Running on $(hostname)"

echo "Python path: $(which python3)"

apptainer_wrapper exec python3 --version



# Run training inside the container

apptainer_wrapper exec python3 /scratch/project_2013587/train_model.py


