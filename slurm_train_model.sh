#!/bin/bash


#SBATCH --account=project_2013501

#SBATCH --partition=gpu

#SBATCH --gres=gpu:v100:4

#SBATCH --cpus-per-task=4

#SBATCH --mem=64G

#SBATCH --time=12:00:00

#SBATCH --output=/scratch/project_2013501/yolo_train.log

#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load required modules

module --force purge

module load python-data
export PYTHONUSERBASE=/scratch/project_2013501/my-python-env
# Print environment and Python version

echo "Running on $(hostname)"

echo "Python path: $(which python3)"

set -xv
python3 $*