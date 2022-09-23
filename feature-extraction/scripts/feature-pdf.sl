#!/bin/bash

#SBATCH -G 0
#SBATCH --mem 8G
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --job-name pdf-features
#SBATCH --output feature-pdf-logs/slurm-%j.out

cd ../src
python feature-pdf.py $1
