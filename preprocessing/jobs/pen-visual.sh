#!/bin/bash

####################################################################
# NAME: pen-visual.sh
# DESCRIPTION: Generate blank images corresponding to pen images for 
#              visualization in QuPath
# POSITIONAL ARGS:
#       1 - WSI Folder
####################################################################

SRC_DIR=$(realpath ../src/)

python $SRC_DIR/make-pen-images.py --threshold 0.5 --wsi $1
