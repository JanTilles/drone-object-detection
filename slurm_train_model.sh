#!/bin/bash


#SBATCH --account=project_xxxxxxx

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
