#!/bin/bash

root_dir=/home/gwinkelmaier/MILKData/NCI-GDC/LGG/Tissue_slide

rm -rf slurm-logs-decomp/*.out

# Loop through Folders
while read folder; do
  # See if predictions folder exists
  if [[ -e $root_dir/$folder/2021 ]]; then
    # If True, submit for decomposition (slurm)
    sbatch decomposition-navab.sl $folder
  else
    # If False, Skip
    continue
  fi
done < file-list-LGG.txt
