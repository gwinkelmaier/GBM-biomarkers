#!/bin/bash

########################################################
# NAME: tile-all.sh
# DESCRIPTION: Tile all available svs files using PyHIST
# PyHIST NOTES:
#    * Output tiles will be at 20x for all WSI
#    * Mask images are generated using 10x magnification
########################################################

# Clean logs
rm slurm-logs/*

# Tile 20x images
while read -r line; do
  if [[ $line != 'folder' ]]; then
    sbatch slurm-files/tile-20x-images.sl $line
  fi
done < lists/20x-list.txt

# Tile 40x images
while read -r line; do
  if [[ $line != 'folder' ]]; then
    sbatch slurm-files/tile-40x-images.sl $line
  fi
done < lists/40x-list.txt
