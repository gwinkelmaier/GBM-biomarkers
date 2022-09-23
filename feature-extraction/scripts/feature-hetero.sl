#!/bin/bash

#SBATCH -G 0
#SBATCH --mem 8G
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 5
#SBATCH --job-name hetero-features
#SBATCH --output feature-hetero-logs/slurm-%j.out

rm feature-hetero-logs/*.out

cd ../src
python feature-hetero.py $1
