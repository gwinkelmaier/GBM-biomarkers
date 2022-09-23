#!/bin/bash

#SBATCH -G 0
#SBATCH --mem 4G
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 5
#SBATCH --job-name tiling
#SBATCH --output slurm-pen-logs/slurm-%j.out
#SBATCH --dependency afterok:63239

##################################################################
#
# NAME: pen-detection.sl
# PURPOSE: Make pen predictions on tiles
#
# NOTES: 
#    * Run with tf_gpu conda env
#
# POSITIONAL ARGS:
#     1 - Folder of WSI
#
##################################################################

PROJECT_HOME=$(realpath ../)

cd $PROJECT_HOME/svm-src
python SVM-predict.py --wsi $1
