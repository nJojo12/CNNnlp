#!/bin/bash
#SBATCH --job-name=Word2Vec
#SBATCH --output=word2vec_output.log
#SBATCH --error=word2vec_err.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --partition=bigbatch


python word2vec.py
