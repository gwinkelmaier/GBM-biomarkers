#!/bin/bash

#SBATCH -G 0
#SBATCH --mem 8G
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --job-name global-pdf
#SBATCH --output global-logs/slurm-%j.out

cd ../src
python feature-global-pdf.py $1 -n 100
