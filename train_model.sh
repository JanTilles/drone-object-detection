#!/bin/bash

#SBATCH --job-name=multi_gpu_training

#SBATCH --account=project_2013587

#SBATCH --partition=gpu

#SBATCH --gres=gpu:v100:4  # Change the number to 1, 2, 3, or 4 as needed

#SBATCH --cpus-per-task=8  # More CPUs for efficient data loading

#SBATCH --mem=48G          # Ensure enough memory

#SBATCH --time=6:00:00

#SBATCH --output=/scratch/project_2013587/tillesja/train_output.log



module purge

module load tensorflow/2.18

module load cuda/12.6.0



apptainer_wrapper exec python3 /scratch/project_2013587/train_model.py


