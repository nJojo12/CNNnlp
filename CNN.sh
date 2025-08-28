#!/bin/bash
#SBATCH --job-name=CNNnlp
#SBATCH --output=cnn_output.log
#SBATCH --error=cnn_error.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --partition=bigbatch


python cnn.py
