#!/bin/bash

#SBATCH -G 0
#SBATCH --mem 4G
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 5
#SBATCH --job-name ot-cluster
#SBATCH --output job-logs/slurm-%j.out

python clustering.py $1 > save_data/$1_logrank_manualBounds.txt
