#!/bin/bash
# 
# Make an inference call on a specific WSI
#
# ex. sbatch infer-wsi.sl 20220110-162436 fe54e5b8-62a9-4432-b46f-70089b621155 save-test
# 

#SBATCH -G 1
#SBATCH --mem 16G
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 9
#SBATCH --job-name wsi-segment

cd ../src/python
python infer-wsi.py $1 $2 $3

python make-prediction-mask.py $2 $1 ${1}-mask
