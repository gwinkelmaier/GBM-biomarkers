#!/bin/bash

#SBATCH -G 0
#SBATCH --mem 8G
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 5
#SBATCH --job-name wsi-feature

cd ../src
python feature-extraction.py $1 -d '20220110-162436'
