#!/bin/bash

#SBATCH -G 0
#SBATCH --mem 8G
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --job-name extended-features
#SBATCH --output extended-feature-logs/slurm-%j.out

cd ../src
python extended-features.py $1
# python area_clip.py $1
