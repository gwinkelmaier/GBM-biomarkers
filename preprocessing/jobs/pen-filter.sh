#!/bin/bash

########################################################
# NAME: pen-filter.sh
# DESCRIPTION: Filter tiles through Pen-detection model
########################################################

# Clean logs
rm slurm-pen-logs/*

# Tile 20x images
while read -r line; do
  if [[ $line != 'folder' ]]; then
    sbatch slurm-files/pen-detection.sl $line
  fi
done < lists/20x-list.txt

# Tile 40x images
while read -r line; do
  if [[ $line != 'folder' ]]; then
    sbatch slurm-files/pen-detection.sl $line
  fi
done < lists/40x-list.txt
