#!/bin/bash

#SBATCH -G 0
#SBATCH --mem 25G
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --job-name tiling
#SBATCH --output slurm-logs/slurm-%j.out

##################################################################
#
# NAME: tile-20x.sl
# PURPOSE: patch WSI tissue into blocks of 224x224 for processing
#
# NOTES: 
#    * Must be run with PyHIST conda env (found inside PyHIST_DIR)
#    * Lowest level magnification is expected to be 20x 
#
# POSITIONAL ARGS:
#     1 - Folder of WSI
#
##################################################################

DATA_DIR=/home/gwinkelmaier/MILKData/NCI-GDC/LGG/Tissue_slide
SVS_FILE=$DATA_DIR/"$1"/*.svs
SAVE_DIR=$(dirname $SVS_FILE)/'2021/STBpatches'

if [[ -d $SAVE_DIR ]]
then
  echo "Save directory set to: $SAVE_DIR"
else
  echo "Creating $SAVE_DIR"
  mkdir $SAVE_DIR
fi


PROJECT_HOME=$(realpath ../..)

PyHIST_DIR=$PROJECT_HOME/PyHIST  # PyHIST LOCATION

cd $PyHIST_DIR
python pyhist.py --patch-size 224 \
                 --output-downsample 1 \
                 --mask-downsample 2 \
                 --tilecross-downsample 4 \
                 --output $SAVE_DIR \
                 --content-threshold 0.05 \
                 --save-tilecrossed-image \
                 --save-mask \
                 --info "verbose" \
                 --save-patches \
                $SVS_FILE
