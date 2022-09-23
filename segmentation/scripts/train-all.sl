#!/bin/bash

#SBATCH -G 2
#SBATCH --mem 16G
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 6
#SBATCH --job-name seg-kfold

cd ../src/python
python train-all.py
