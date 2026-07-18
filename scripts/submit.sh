#!/bin/bash
#SBATCH --job-name=spleen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/ori.grossman/nn_final/experiments/slurm_%j.out

CONTAINER=/opt/containers/pytorch-25.04.sif
cd /home/ori.grossman/nn_final
apptainer exec --nv "$CONTAINER" python notebooks/train.py "$@"
